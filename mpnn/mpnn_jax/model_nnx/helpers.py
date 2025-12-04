import os
import shutil
from typing import Callable
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx import RngKey
import jax
import jax.numpy as jnp


def _absdir(path: str) -> str:
    apath = os.path.abspath(path)
    os.makedirs(os.path.dirname(apath), exist_ok=True)
    return apath

def save_checkpoint(ckpt_dir: str, model: nnx.Module, W_mean: float, W_std: float) -> str:
    """
    Save an NNX model state to a directory checkpoint.
    """
    ckpt_dir = _absdir(ckpt_dir)
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    _, state = nnx.split(model)
    state = _encode_rng_state(state)

    checkpointer = ocp.StandardCheckpointer()
    payload = {
        "state": state,
        "W_mean": float(W_mean), 
        "W_std": float(W_std), 
    }

    checkpointer.save(ckpt_dir, payload, force=True)
    checkpointer.wait_until_finished()
    return ckpt_dir

def _encode_rng_state(state: nnx.State) -> nnx.State:
    """
    Replace PRNGKey-like leaves in NNX state with raw uint32 key data.
    """
    flat = nnx.to_flat_state(state)
    encoded = []
    for path, leaf in flat:
        if isinstance(leaf, nnx.VariableState):
            value = leaf.value
            if isinstance(value, jax.Array) and value.dtype == jax.dtypes.prng_key:
                leaf = leaf.replace(jax.random.key_data(value))
        encoded.append((path, leaf))
    return nnx.from_flat_state(encoded, cls=type(state))

def _decode_rng_state(state: nnx.State) -> nnx.State:
    """
    Convert raw uint32 key data back into JAX PRNGKey variables.
    """
    flat = nnx.to_flat_state(state)
    decoded = []
    for path, leaf in flat:
        if isinstance(leaf, nnx.VariableState):
            value = leaf.value
            if (
                isinstance(value, jax.Array)
                and value.dtype == jnp.uint32
                and issubclass(leaf.type, RngKey)
            ):
                leaf = leaf.replace(jax.random.wrap_key_data(value))
        decoded.append((path, leaf))
    return nnx.from_flat_state(decoded, cls=type(state))

def load_checkpoint(ckpt_dir: str, model_ctor: Callable[[], nnx.Module]):
    """
    Restore an NNX model from a directory checkpoint.
    """
    ckpt_dir = _absdir(ckpt_dir)

    abstract_model = nnx.eval_shape(model_ctor)
    graphdef, abstract_state = nnx.split(abstract_model)
    abstract_state = _encode_rng_state(abstract_state)

    template = {
        "state": abstract_state, 
        "W_mean": 0.0, 
        "W_std": 1.0
    }

    checkpointer = ocp.StandardCheckpointer()
    payload = checkpointer.restore(ckpt_dir, target=template)

    state = nnx.State(payload["state"])
    state = _decode_rng_state(state)
    model = nnx.merge(graphdef, state)

    return {
        "model": model,
        "W_mean": float(payload["W_mean"]),
        "W_std": float(payload["W_std"]),
    }