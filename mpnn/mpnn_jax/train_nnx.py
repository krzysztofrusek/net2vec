import jax
import jax.numpy as jnp
import optax
from loader import load_npz_as_graphs, make_loader
from metrics import mae, r2, mse
from flax import nnx
from model_nnx.model import MPNN_NNX
from tqdm import tqdm, trange
from model_nnx.helpers import save_checkpoint
from config import ModelConfig, TrainingConfig, get_norm
import numpy as np

def _is_trainable(leaf):
    """
    Decide whether a leaf from an NNX state PyTree is trainable.
    """
    if isinstance(leaf, (jax.Array, jnp.ndarray, np.ndarray)):
        return jnp.issubdtype(leaf.dtype, jnp.inexact)
    return isinstance(leaf, float)

def partition_state(state):
    """
    Split an NNX `state` into three aligned PyTrees:
      - `trainable`: leaves that will be optimized (others replaced by dummy scalars),
      - `frozen`:    non-trainable leaves kept as-is (trainables set to None),
      - `mask`:      boolean structure indicating which leaves are trainable.

    This allows using `optax.masked` to update only trainable leaves, while still being able to reconstruct the full model state later.
    """
    def make_trainable(leaf):
        return leaf if _is_trainable(leaf) else jnp.zeros((), dtype=jnp.float32)
    def make_frozen(leaf):
        return None if _is_trainable(leaf) else leaf
    def make_mask(leaf):
        return bool(_is_trainable(leaf))
    trainable = jax.tree_util.tree_map(make_trainable, state)
    frozen = jax.tree_util.tree_map(make_frozen, state)
    mask = jax.tree_util.tree_map(make_mask, state)
    return trainable, frozen, mask

def combine_state(trainable, frozen, mask):
    """
    Merge `trainable` and `frozen` PyTrees back into a full state using `mask`.
    """
    def select(t, f, m): return t if m else f
    return jax.tree_util.tree_map(select, trainable, frozen, mask)

def build_apply_fn(graphdef, frozen_state, mask):    
    """
    Create a pure apply function that reconstructs an NNX module from:
      - static graph definition (`graphdef`),
      - partial state composed of current `params` + `frozen_state`.
    """
    def apply_fn(params, batch_graph):
        full_state = combine_state(params, frozen_state, mask)
        model = nnx.merge(graphdef, full_state)
        return model(batch_graph)  # batch_graph = GraphsTuple
    return apply_fn

def build_loss_fn(apply_fn):    
    """
    Wrap the forward pass with MSE loss in normalized label space.
    """
    def loss_fn(params, batch_graph):
        preds = apply_fn(params, batch_graph)
        labels = batch_graph.globals.squeeze()

        preds = preds[:-1]
        labels = labels[:-1]
        
        loss = jnp.mean((preds - labels) ** 2)
        return loss, preds
    return jax.jit(loss_fn)

def build_train_step(loss_fn, optimizer):
    """
    Create a JIT-compiled single training step for masked optimization.
    """
    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return train_step

if __name__ == "__main__":
    cfg = TrainingConfig()
    model_cfg = ModelConfig()
    norm = get_norm(cfg.norm_profile)

    graphs = load_npz_as_graphs(cfg.train_dataset_path, norm)
    eval_graphs = load_npz_as_graphs(cfg.val_dataset_path, norm)

    W_mean = float(norm.W_shift)
    W_std  = float(norm.W_scale)

    key = jax.random.key(cfg.seed)
    BATCH_SIZE = cfg.batch_size
    train_loader = make_loader(graphs, BATCH_SIZE, key)
    val_loader = make_loader(eval_graphs, BATCH_SIZE, key)

    example_graph, key = next(train_loader)
    rngs = nnx.Rngs(params=key)
    model = MPNN_NNX(
        hidden_dim=model_cfg.hidden_dim,
        N_H=model_cfg.N_H,
        rn=model_cfg.rn,
        num_passes=model_cfg.num_passes,
        rngs=rngs,
    )
    graphdef, params = nnx.split(model)
    params, frozen_state, mask = partition_state(params)

    base_optimizer = optax.rmsprop(learning_rate=cfg.learning_rate)
    optimizer = optax.masked(base_optimizer, mask)
    opt_state = optimizer.init(params)

    apply_fn = build_apply_fn(graphdef, frozen_state, mask)
    loss_fn = build_loss_fn(apply_fn)
    train_step = build_train_step(loss_fn, optimizer)

    best_val_mse = float("inf")
    no_improve_checks = 0
    tolerance = 1e-6
    best_params = params
    best_step = 0
    best_ckpt_path = f"{cfg.output_path}/model_nnx_best"

    try:
        for step in trange(cfg.steps, desc="Model training"):
            batch_graph, key = next(train_loader)
            params, opt_state, loss = train_step(params, opt_state, batch_graph)

            if step % cfg.log_interval == 1:
                preds  = apply_fn(params, batch_graph)
                labels = batch_graph.globals.squeeze()

                mse_val = mse(labels, preds)
                mae_val = mae(labels, preds)
                r2_val  = r2(labels, preds)

                tqdm.write(f"[TRAIN step {step:05d}] MSE={mse_val:.6f}  MAE={mae_val:.6f}  R²={r2_val:.6f}")

                val_graph, key = next(val_loader)
                val_preds  = apply_fn(params, val_graph)
                val_labels = val_graph.globals.squeeze()

                val_mse = mse(val_labels, val_preds)
                val_mae = mae(val_labels, val_preds)
                val_r2  = r2(val_labels, val_preds)

                tqdm.write(f"[VAL step {step:05d}] MSE={val_mse:.6f}  MAE={val_mae:.6f}  R²={val_r2:.6f}")
                tqdm.write(f"step {step:04d}   loss = {loss:.6f}")

                val_mse_float = float(val_mse)
                if val_mse_float + tolerance < best_val_mse:
                    
                    best_val_mse = val_mse_float
                    no_improve_checks = 0

                    best_params = params
                    best_step = step

                else:
                    if cfg.use_early_stopping and cfg.early_stopping_patience > 0:
                        no_improve_checks += 1

                        if no_improve_checks >= cfg.early_stopping_patience:
                            tqdm.write(
                                f"Early stopping at step {step:05d}. "
                                f"Best VAL MSE: {best_val_mse:.6f}"
                                f"(step {best_step:05d})"
                            )
                            break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        full_state = combine_state(params, frozen_state, mask)
        model_to_save = nnx.merge(graphdef, full_state)
        save_checkpoint(f"{cfg.output_path}/model_nnx_checkpoint", model_to_save, W_mean, W_std)
        raise

    full_state_best = combine_state(best_params, frozen_state, mask)
    model_best = nnx.merge(graphdef, full_state_best)
    tqdm.write(
        f"Saving BEST params from step {best_step:05d} with "
        f"VAL MSE={best_val_mse:.6f} -> {best_ckpt_path}"
    )
    save_checkpoint(best_ckpt_path, model_best, W_mean, W_std)

    full_state_final = combine_state(params, frozen_state, mask)
    model_final = nnx.merge(graphdef, full_state_final)
    save_checkpoint(
        f"{cfg.output_path}/model_nnx_final",
        model_final,
        W_mean,
        W_std,
    )
