import jax
import jax.numpy as jnp
import numpy as np
import jraph
from config import NormConfig


def _nearest_bigger_power_of_two(x: int) -> int:
    """
    Computes the nearest power of two greater than x for padding.
    """
    y = 2
    while y < x:
        y *= 2
    return y

def sample_to_jraph(mu, L, R, W, norm: NormConfig):
    """
    Convert a single synthetic queueing-network sample (μ, Λ, R, W) into a
    `jraph.GraphsTuple` used by the MPNN.
    """
    mu = jnp.asarray(np.asarray(mu, np.float32))
    L  = jnp.asarray(np.asarray(L, np.float32))
    R  = jnp.asarray(np.asarray(R, np.float32))
    W  = jnp.asarray(W)

    mu_norm = (mu - norm.mu_shift) / norm.mu_scale
    W_norm  = (W  - norm.W_shift) / norm.W_scale

    nodes = jnp.concatenate([mu_norm, L], axis=1)                
    senders, receivers = jnp.where(R > 0)
    edges = R[senders, receivers].reshape(-1, 1)

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders.astype(jnp.int32),
        receivers=receivers.astype(jnp.int32),
        globals=jnp.array([W_norm], dtype=jnp.float32),
        n_node=jnp.array([nodes.shape[0]]),
        n_edge=jnp.array([edges.shape[0]])
    )


def load_npz_as_graphs(path, norm: NormConfig):
    """
    Load a dataset of samples from a compressed NPZ file and convert each
    sample to a `jraph.GraphsTuple`.
    """

    data = np.load(path, allow_pickle=True)

    mus = np.array(data["mu"])
    Ls  = data["L"]
    Rs  = data["R"]
    Ws  = data["W"]
    return [sample_to_jraph(mu, L, R, W, norm) for mu, L, R, W in zip(mus, Ls, Rs, Ws)]


def make_loader(graphs, batch_size, key):
    """
    Build an infinite randomized loader that:
      1. precomputes all mini-batches via `jraph.batch`,
      2. pads each batched graph once with `jraph.pad_with_graphs` to have
         static shapes (power-of-two sizes),
      3. shuffles the order of these precomputed batches on each iteration
         and yields an infinite stream of padded `GraphsTuple`s and updated keys.
    """
    num_graphs = len(graphs)
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, num_graphs)
    perm = perm[: (num_graphs // batch_size) * batch_size].reshape(-1, batch_size)

    batches = []
    for row in perm:
        batch = jraph.batch([graphs[int(i)] for i in np.array(row)])

        pad_nodes_to = _nearest_bigger_power_of_two(int(jnp.sum(batch.n_node))) + 1
        pad_edges_to = _nearest_bigger_power_of_two(int(jnp.sum(batch.n_edge)))
        pad_graphs_to = batch.n_node.shape[0] + 1

        padded_batch = jraph.pad_with_graphs(
            batch,
            n_node=pad_nodes_to,
            n_edge=pad_edges_to,
            n_graph=pad_graphs_to,
        )
        batches.append(padded_batch)

    while True:
        key, subkey = jax.random.split(key)
        order = jax.random.permutation(subkey, len(batches))
        for j in order:
            yield batches[int(j)], key
