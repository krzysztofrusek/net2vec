import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph


def init_node_state(node_features, N_PAD):
    """
    Initialize hidden node state by padding input features with zeros.
    """
    N = node_features.shape[0]
    pad = jnp.zeros((N, N_PAD), dtype=node_features.dtype)
    return jnp.concatenate([node_features, pad], axis=1)

def graph_indices_from_n_node(graph: jraph.GraphsTuple) -> jnp.ndarray:
    """
    Compute graph indices for segment operations over a batched GraphsTuple.
    """
    B = graph.n_node.shape[0]
    total_nodes = graph.nodes.shape[0]
    return jnp.repeat(
        jnp.arange(B, dtype=jnp.int32),
        graph.n_node,
        total_repeat_length=total_nodes,
    )

class MessageFunction(nn.Module):
    """
    Computes messages for each edge based on the sender node state and the edge feature.
    """
    hidden_dim: int
    N_H: int

    @nn.compact
    def __call__(self, h_i, e_ij):
        A = nn.Dense(self.hidden_dim)(e_ij)
        A = nn.selu(A)
        A = nn.Dense(self.N_H * self.N_H)(A)
        A = A.reshape((-1, self.N_H, self.N_H))

        b = nn.Dense(self.hidden_dim)(e_ij)
        b = nn.selu(b)
        b = nn.Dense(self.N_H)(b)

        m_ij = jnp.einsum("eij,ej->ei", A, h_i) + b

        return m_ij


class MessagePassingLayer(nn.Module):
    """
    One message-passing round: gather -> edge messages -> receiver sum.
    """
    hidden_dim: int
    N_H: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        """
        graph.nodes:     (N_total, N_H)
        graph.edges:     (E_total, 1)
        graph.senders:   (E_total,)
        graph.receivers: (E_total,)
        """

        h = graph.nodes               # (N, N_H)
        e = graph.edges               # (E, 1)
        senders = graph.senders       # (E,)
        receivers = graph.receivers   # (E,)

        h_i = h[senders]              # (E, N_H)

        m_ij = MessageFunction(self.hidden_dim, self.N_H)(h_i, e)   # (E, N_H)

        edge_mask = jraph.get_edge_padding_mask(graph).astype(m_ij.dtype)
        edge_mask = edge_mask.reshape((-1, 1))
        m_ij = m_ij * edge_mask

        m_j = jax.ops.segment_sum(m_ij, receivers, num_segments=h.shape[0])   # (N, N_H)

        return m_j

class NodeUpdateLayer(nn.Module):
    """
    Node state update via a GRUCell(h_old, m_j).
    """
    N_H: int   

    @nn.compact
    def __call__(self, h_old, m_j):
        """
        h_old: (N, N_H)
        m_j:   (N, N_H)  
        returns: h_new: (N, N_H)
        """
        gru = nn.GRUCell(self.N_H)

        h_new, _ = gru(h_old, m_j)   
        return h_new
    
class ReadoutLayer(nn.Module):    
    """
    Graph-level readout with gating: sigmoid(i(h,x)) ⊙ j(h,x), pooled by sum.
    """
    rn: int      
    N_H: int     

    @nn.compact
    def __call__(self, graph):
        h = graph.nodes          # (N, N_H)
        x = graph.nodes[:, :2]   
        hx = jnp.concatenate([h, x], axis=-1)  # (N, N_H+2)

        i_out = nn.tanh(nn.Dense(self.rn)(hx))
        i_out = nn.Dense(self.rn)(i_out)

        j_out = nn.selu(nn.Dense(self.rn)(hx))
        j_out = nn.Dense(self.rn)(j_out)

        RR = nn.sigmoid(i_out) * j_out   # (N, rn)

        node_mask = jraph.get_node_padding_mask(graph).astype(RR.dtype)
        node_mask = node_mask.reshape((-1, 1))
        RR = RR * node_mask

        graph_idx = graph_indices_from_n_node(graph)
        pooled = jax.ops.segment_sum(RR, graph_idx, num_segments=graph.n_node.shape[0])
        out = nn.selu(nn.Dense(self.rn)(pooled))
        out = nn.Dense(1)(out)     

        return out.squeeze()       # (B,)
    
class MPNN(nn.Module):
    """
    Full Message Passing Neural Network:
      1. Initialize node hidden state h0 = concat([x, zeros])  so that dim = N_H.
      2. Apply `num_passes` rounds of message passing + GRU update.
      3. Apply gated readout and regress a graph-level scalar.
    """
    hidden_dim: int   # args.Mhid (e.g. 8)
    N_H: int          # 2 + N_PAD = e.g. 14
    rn: int           # args.rn (e.g. 8)
    num_passes: int   # args.N_PAS (e.g. 4)

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        h0 = init_node_state(graph.nodes, self.N_H - 2)
        graph = graph._replace(nodes=h0)
        message_passing_layer = MessagePassingLayer(self.hidden_dim, self.N_H)
        node_update_layer = NodeUpdateLayer(self.N_H)

        for _ in range(self.num_passes):
            m_j = message_passing_layer(graph)
            h_old = graph.nodes
            h_new = node_update_layer(h_old, m_j)
            graph = graph._replace(nodes=h_new)

        return ReadoutLayer(self.rn, self.N_H)(graph)
