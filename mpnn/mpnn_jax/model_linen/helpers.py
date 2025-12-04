import jax
import jax.numpy as jnp
import os
import orbax.checkpoint as ocp
import jraph


def _nearest_bigger_power_of_two(x: int) -> int:
    """
    Computes the nearest power of two greater than x for padding.
    """
    y = 2
    while y < x:
        y *= 2
    return y


def _absdir(path: str) -> str:
    """Resolve an absolute path and ensure its parent directory exists."""
    apath = os.path.abspath(path)
    os.makedirs(os.path.dirname(apath), exist_ok=True)
    return apath


def save_checkpoint(ckpt_dir: str, params, W_mean: float, W_std: float) -> str:
    """
    Save a Linen checkpoint using Orbax.
    """
    ckpt_dir = _absdir(ckpt_dir)

    payload = {
        "params": params,
        "W_mean": float(W_mean),
        "W_std": float(W_std),
    }

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir, payload, force=True)
    checkpointer.wait_until_finished()



def load_checkpoint(ckpt_dir: str, model, example_graph):
    """
    Restore a Linen checkpoint using Orbax.
    """
    ckpt_dir = _absdir(ckpt_dir)

    n = int(example_graph.n_node[0])
    e = int(example_graph.n_edge[0])

    pad_nodes_to = _nearest_bigger_power_of_two(n) + 1
    pad_edges_to = _nearest_bigger_power_of_two(e)
    pad_graphs_to = example_graph.n_node.shape[0] + 1

    padded_example = jraph.pad_with_graphs(
        example_graph,
        n_node=pad_nodes_to,
        n_edge=pad_edges_to,
        n_graph=pad_graphs_to,
    )

    params_template = model.init(jax.random.key(0), padded_example)

    template = {
        "params": params_template,
        "W_mean": 0.0,
        "W_std": 1.0,
    }

    checkpointer = ocp.StandardCheckpointer()
    payload = checkpointer.restore(ckpt_dir, target=template)
    return payload
