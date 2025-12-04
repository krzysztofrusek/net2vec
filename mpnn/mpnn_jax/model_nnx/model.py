from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import jraph
from flax import nnx
from jax import nn as jnn
import jax.ops as jops
import jraph

def _init_node_state(node_features: jnp.ndarray, N_PAD: int) -> jnp.ndarray:
    """
    Initialize hidden node states by padding input features with zeros.
    """
    N = node_features.shape[0]
    pad = jnp.zeros((N, N_PAD), dtype=node_features.dtype)
    return jnp.concatenate([node_features, pad], axis=1)

def _graph_indices_from_n_node(graph: jraph.GraphsTuple) -> jnp.ndarray:
    """
    Create a per-node graph index vector for batched graphs.
    """
    B = graph.n_node.shape[0]
    total_nodes = graph.nodes.shape[0]
    return jnp.repeat(
        jnp.arange(B, dtype=jnp.int32),
        graph.n_node,
        total_repeat_length=total_nodes,
    )

class MessageFunction(nnx.Module):
    """
    Computes messages for each edge based on the sender node state and the edge feature.
    """
    hidden_dim: int
    N_H: int

    def __init__(self, *, hidden_dim: int, N_H: int, rngs: Optional[nnx.Rngs] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.N_H = N_H
        self.lin_A1 = nnx.Linear(1, hidden_dim, rngs=rngs)
        self.lin_A2 = nnx.Linear(hidden_dim, N_H * N_H, rngs=rngs)
        self.lin_b1 = nnx.Linear(1, hidden_dim, rngs=rngs)
        self.lin_b2 = nnx.Linear(hidden_dim, N_H, rngs=rngs)

    def __call__(self, h_i: jnp.ndarray, e_ij: jnp.ndarray) -> jnp.ndarray:
        # A(e)
        A = self.lin_A1(e_ij)
        A = jnn.selu(A)
        A = self.lin_A2(A).reshape((-1, self.N_H, self.N_H))
        # b(e)
        b = self.lin_b1(e_ij)
        b = jnn.selu(b)
        b = self.lin_b2(b)
        # m_ij
        m_ij = jnp.einsum("eij,ej->ei", A, h_i) + b
        return m_ij

class MessagePassingLayer(nnx.Module):
    """
    One message-passing round: gather -> edge messages -> receiver sum.
    """
    hidden_dim: int
    N_H: int

    def __init__(self, *, hidden_dim: int, N_H: int, rngs: Optional[nnx.Rngs] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.N_H = N_H
        self.msg = MessageFunction(hidden_dim=hidden_dim, N_H=N_H, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        h = graph.nodes               # (N, N_H)
        e = graph.edges               # (E, 1)
        senders = jnp.asarray(graph.senders, dtype=jnp.int32)     # (E,)
        receivers = jnp.asarray(graph.receivers, dtype=jnp.int32) # (E,)

        h_i = jnp.take(h, senders, axis=0)  # (E, N_H)
        m_ij = self.msg(h_i, e)             # (E, N_H)

        edge_mask = jraph.get_edge_padding_mask(graph).astype(m_ij.dtype)
        edge_mask = edge_mask.reshape((-1, 1))
        m_ij = m_ij * edge_mask

        m_j = jops.segment_sum(m_ij, receivers, num_segments=h.shape[0])  # (N, N_H)
        return m_j

class NodeUpdateLayer(nnx.Module):
    """
    Node-state update via a GRU cell: h_new = GRU(h_old, m_j).
    """
    N_H: int

    def __init__(self, *, N_H: int, rngs: Optional[nnx.Rngs] = None):
        super().__init__()
        self.N_H = N_H
        self.gru = nnx.GRUCell(N_H, N_H, rngs=rngs)

    def __call__(self, h_old: jnp.ndarray, m_j: jnp.ndarray) -> jnp.ndarray:
        out = self.gru(h_old, m_j)
        return out[0] if isinstance(out, tuple) else out

class ReadoutLayer(nnx.Module):
    """
    Readout over nodes to predict W per-graph.
    """
    rn: int
    N_H: int

    def __init__(self, *, rn: int, N_H: int, rngs: Optional[nnx.Rngs] = None):
        super().__init__()
        self.rn = rn
        self.N_H = N_H
        # i-path
        self.i1 = nnx.Linear(N_H + 2, rn, rngs=rngs)
        self.i2 = nnx.Linear(rn, rn, rngs=rngs)
        # j-path
        self.j1 = nnx.Linear(N_H + 2, rn, rngs=rngs)
        self.j2 = nnx.Linear(rn, rn, rngs=rngs)
        # head
        self.h1 = nnx.Linear(rn, rn, rngs=rngs)
        self.h2 = nnx.Linear(rn, 1, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple):
        h = graph.nodes            # (N, N_H)
        x = graph.nodes[:, :2]     
        hx = jnp.concatenate([h, x], axis=-1)  # (N, N_H+2)

        i_out = jnp.tanh(self.i1(hx))
        i_out = self.i2(i_out)

        j_out = jnn.selu(self.j1(hx))
        j_out = self.j2(j_out)

        RR = jnn.sigmoid(i_out) * j_out  # (N, rn)
        node_mask = jraph.get_node_padding_mask(graph).astype(RR.dtype)
        node_mask = node_mask.reshape((-1, 1))
        RR = RR * node_mask

        graph_idx = _graph_indices_from_n_node(graph)
        pooled = jops.segment_sum(RR, graph_idx, num_segments=graph.n_node.shape[0])  # (B, rn)

        out = jnn.selu(self.h1(pooled))
        out = self.h2(out)  # (B,1)
        out = out.squeeze(-1)  # (B,)

        return out

class MPNN_NNX(nnx.Module):
    """
    End-to-end MPNN in NNX: message passing + GRU updates + readout.
    """
    hidden_dim: int
    N_H: int
    rn: int
    num_passes: int

    def __init__(self, *, hidden_dim: int, N_H: int, rn: int, num_passes: int, rngs: nnx.Rngs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.N_H = N_H
        self.rn = rn
        self.num_passes = num_passes

        self.mp = MessagePassingLayer(hidden_dim=hidden_dim, N_H=N_H, rngs=rngs)
        self.upd = NodeUpdateLayer(N_H=N_H, rngs=rngs)
        self.readout = ReadoutLayer(rn=rn, N_H=N_H, rngs=rngs)

    def __call__(self, graph: jraph.GraphsTuple):
        h0 = _init_node_state(graph.nodes, self.N_H - 2)
        graph = graph._replace(nodes=h0)

        for _ in range(self.num_passes):
            m_j = self.mp(graph)
            h_new = self.upd(graph.nodes, m_j)
            graph = graph._replace(nodes=h_new)

        return self.readout(graph)