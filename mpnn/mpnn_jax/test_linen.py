import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from model_linen.model import MPNN
from model_linen.helpers import load_checkpoint
import loader
from metrics import mse, r2, pearson
from config import get_norm
from loader import make_loader
import jax


def evaluate_model(params, model, graphs, batch_size=64, seed=0):
    """
    Run batched evaluation on a list of graphs.
    """
    key = jax.random.key(seed)
    test_loader = make_loader(graphs, batch_size, key)

    preds_list = []
    labels_list = []

    num_batches = (len(graphs) + batch_size - 1) // batch_size
    for _ in range(num_batches):

        batch_graph, key = next(test_loader)
        batch_preds  = model.apply(params, batch_graph).squeeze()
        batch_labels = batch_graph.globals.squeeze()

        batch_preds = batch_preds[:-1]
        batch_labels = batch_labels[:-1]

        preds_list.append(batch_preds)
        labels_list.append(batch_labels)

    preds = jnp.concatenate(preds_list, axis=0)
    labels = jnp.concatenate(labels_list, axis=0)

    return preds, labels


def main():
    """
    Entry point for evaluation:
      1. Load the test split and construct the MPNN model.
      2. Restore parameters from a checkpoint (shapes bootstrapped from a sample graph).
      3. Evaluate the model (normalized space) and report MSE, R², Pearson.
      4. Produce diagnostic plots: prediction scatter and residual histogram.
    """  
    from config import TestConfig, ModelConfig
    test_cfg, model_cfg = TestConfig(), ModelConfig()
    norm = get_norm(test_cfg.norm_profile)
    os.makedirs(test_cfg.output_path, exist_ok=True)

    print("Loading test dataset:", test_cfg.test_dataset_path)
    test_graphs = loader.load_npz_as_graphs(test_cfg.test_dataset_path, norm)

    model = MPNN(
        hidden_dim=model_cfg.hidden_dim,
        N_H=model_cfg.N_H,
        rn=model_cfg.rn,
        num_passes=model_cfg.num_passes,
    )


    print("Loading checkpoint:", test_cfg.checkpoint_path)
    payload = load_checkpoint(test_cfg.checkpoint_path, model, test_graphs[0])
    params = payload["params"]
    W_mean = payload["W_mean"]
    W_std = payload["W_std"]

    preds, labels = evaluate_model(params, model, test_graphs)


    mse_norm_val = float(mse(labels, preds))
    r2_val = float(r2(labels, preds))
    pearson_val = float(pearson(labels, preds))

    print(f"MSE: {mse_norm_val:.17f} R**2: {r2_val:.17f} Pearson: {pearson_val:.17f}")

    # Plot prediction scatter
    plt.figure()
    plt.plot(labels, preds, ".", alpha=0.6)
    lims = [min(labels.min(), preds.min()), max(labels.max(), preds.max())]
    plt.plot(lims, lims, "k-", alpha=0.75)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.title("Evaluation")
    plt.grid(True)
    plt.savefig(f"{test_cfg.output_path}/eval.svg", format="svg")
    plt.close()

    # Histogram residuals
    plt.figure()
    plt.hist((labels - preds), bins=50)
    plt.title("Histogram of residuals")
    plt.grid(True)
    plt.savefig(f"{test_cfg.output_path}/residuals_hist.svg", format="svg")
    plt.close()

    print(f"Saved evaluation plots to: {test_cfg.output_path}")


if __name__ == "__main__":
    main()
