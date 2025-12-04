import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from flax import nnx
from model_nnx.model import MPNN_NNX
from model_nnx.helpers import load_checkpoint
import loader
from loader import make_loader
from metrics import mse, r2, pearson
from config import TestConfig, ModelConfig, get_norm

def evaluate_model(model_apply, graphs, batch_size=64, seed=0):    
    """
    Run batched evaluation for an NNX MPNN using a callable `model_apply`.

    model_apply:
        Function taking a batch `(graph, node_mask, edge_mask)` and returning
        predictions of shape (B,) in the normalized W space.
    """

    key = jax.random.key(seed)
    test_loader = make_loader(graphs, batch_size, key)

    preds_list, labels_list = [], []
    num_batches = (len(graphs) + batch_size - 1) // batch_size
    for _ in range(num_batches):
        batch_graph, key = next(test_loader)
        batch_preds  = model_apply(batch_graph).squeeze()
        batch_labels = batch_graph.globals.squeeze()
        preds_list.append(batch_preds); labels_list.append(batch_labels)

    preds  = jnp.concatenate(preds_list, axis=0)
    labels = jnp.concatenate(labels_list, axis=0)
    return preds, labels

def main():
    """
    Offline evaluation entry point for the NNX variant:
      1. Load test graphs and construct an abstract MPNN_NNX ctor.
      2. Restore a serialized NNX state via `orbax`-based helper.
      3. Evaluate predictions and report MSE, R², Pearson in the normalized space.
      4. Save diagnostic plots (scatter + residual histogram).
    """

    test_cfg, model_cfg = TestConfig(), ModelConfig()
    norm = get_norm(test_cfg.norm_profile)
    os.makedirs(test_cfg.output_path, exist_ok=True)

    print("Loading test dataset:", test_cfg.test_dataset_path)
    test_graphs = loader.load_npz_as_graphs(test_cfg.test_dataset_path, norm)

    ctor = lambda: MPNN_NNX(
        hidden_dim=model_cfg.hidden_dim,
        N_H=model_cfg.N_H,
        rn=model_cfg.rn,
        num_passes=model_cfg.num_passes,
        rngs=nnx.Rngs(params=jax.random.key(0))
    )

    print("Loading checkpoint:", test_cfg.checkpoint_path)
    payload = load_checkpoint(os.path.abspath(test_cfg.checkpoint_path), ctor)
    model = payload["model"] 
    print("Checkpoint restored.")

    model_apply = lambda batch: model(batch)

    preds, labels = evaluate_model(model_apply, test_graphs)

    mse_val = float(mse(labels, preds))
    r2_val = float(r2(labels, preds))
    pearson_val = float(pearson(labels, preds))
    print(f"MSE: {mse_val:.17f} R**2: {r2_val:.17f} Pearson: {pearson_val:.17f}")

    # Plot prediction scatter
    plt.figure()
    plt.plot(labels, preds, ".", alpha=0.6)
    lims = [min(float(labels.min()), float(preds.min())),
            max(float(labels.max()), float(preds.max()))]
    plt.plot(lims, lims, "k-", alpha=0.75)
    plt.xlabel("True (normalized W)")
    plt.ylabel("Predicted (normalized W)")
    plt.title("Evaluation (NNX parity with Linen)")
    plt.grid(True)
    plt.savefig(f"{test_cfg.output_path}/eval.pdf")
    plt.close()

    # Histogram residuals
    plt.figure()
    plt.hist((labels - preds), bins=50)
    plt.title("Residuals (NNX)")
    plt.grid(True)
    plt.savefig(f"{test_cfg.output_path}/residuals_hist.pdf")
    plt.close()

    print(f"Saved eval.pdf and residuals_hist.pdf to {test_cfg.output_path}")

if __name__ == "__main__":
    main()
