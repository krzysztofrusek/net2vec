import jax
import jax.numpy as jnp
from loader import load_npz_as_graphs, make_loader
import optax
from model_linen.model import MPNN
from metrics import mae, r2, mse
from config import TrainingConfig, ModelConfig, get_norm
from model_linen.helpers import save_checkpoint
from tqdm import trange, tqdm


model_cfg = ModelConfig()
model = MPNN(
    hidden_dim=model_cfg.hidden_dim,
    N_H=model_cfg.N_H,
    rn=model_cfg.rn,
    num_passes=model_cfg.num_passes
)


@jax.jit
def loss_fn(params, batch):
    """
    Compute the MSE loss.
    """
    preds = model.apply(params, batch)
    labels = batch.globals.squeeze()
    preds = preds[:-1]
    labels = labels[:-1]
    loss = jnp.mean((preds - labels)**2)
    return loss, preds


@jax.jit 
def train_step(params, opt_state, batch):
    """
    One optimization step: dLoss/dθ, optimizer update, new params/state.
    """

    (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


if __name__ == "__main__":
    """
    Training script for the Linen implementation of the MPNN:
    1. Load datasets and normalize features/labels according to `norm_profile`.
    2. Build a Linen MPNN model and initialize parameters.
    3. Train with RMSProp in the normalized space; periodically log metrics on train and validation batches.
    4. Optionally use early stopping based on validation MSE.
    5. Save best and final checkpoints using Orbax.
    """

    cfg = TrainingConfig()
    norm = get_norm(cfg.norm_profile)

    graphs = load_npz_as_graphs(cfg.train_dataset_path, norm)
    eval_graphs = load_npz_as_graphs(cfg.val_dataset_path, norm)

    W_mean = float(norm.W_shift)
    W_std  = float(norm.W_scale)

    key = jax.random.key(cfg.seed)

    BATCH_SIZE = cfg.batch_size
    train_loader = make_loader(graphs, BATCH_SIZE, key)
    val_loader = make_loader(eval_graphs, BATCH_SIZE, key)

    batch_graph, key = next(train_loader)
    params = model.init(key, batch_graph)

    optimizer = optax.rmsprop(learning_rate=cfg.learning_rate)
    opt_state = optimizer.init(params)

    best_val_mse = float("inf")
    no_improve_checks = 0
    tolerance = 1e-6

    best_params = params  
    best_step = 0          
    best_ckpt_path = f"{cfg.output_path}/model_linen_best"

    try:
        for step in trange(cfg.steps, desc="Model training"):
            batch_graph, key = next(train_loader)
            params, opt_state, loss = train_step(params, opt_state, batch_graph)

            if step % cfg.log_interval == 1:
                preds  = model.apply(params, batch_graph)
                labels = batch_graph.globals.squeeze()

                mse_val = mse(labels, preds)
                mae_val = mae(labels, preds)
                r2_val = r2(labels, preds)

                tqdm.write(f"[TRAIN step {step:05d}] "
                    f"MSE={mse_val:.6f}  "
                    f"MAE={mae_val:.6f}  "
                    f"R²={r2_val:.6f}")
                
                val_graph, key = next(val_loader)
                val_preds  = model.apply(params, val_graph)
                val_labels = val_graph.globals.squeeze()

                val_mse = mse(val_labels, val_preds)
                val_mae = mae(val_labels, val_preds)
                val_r2  = r2(val_labels, val_preds)

                tqdm.write(f"[VAL step {step:05d}] "
                        f"MSE={val_mse:.6f}  MAE={val_mae:.6f}  R²={val_r2:.6f}")
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
        print("\n Training interrupted by user. Saving checkpoint...")
        save_checkpoint(f"{cfg.output_path}/model_linen_checkpoint", params, W_mean, W_std)
        raise
    
    tqdm.write(
    f"Saving BEST params from step {best_step:05d} with "
    f"VAL MSE={best_val_mse:.6f} -> {best_ckpt_path}"
    )
    
    save_checkpoint(best_ckpt_path, best_params, W_mean, W_std)
    save_checkpoint(f"{cfg.output_path}/model_linen_final", params, W_mean, W_std)
