from typing import Dict, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt

def _to_numpy(x) -> np.ndarray:
    """Convert torch / numpy embedding to 1D numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).squeeze().reshape(-1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D arrays."""
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def _z_norm(x: np.ndarray) -> np.ndarray:
    """Z-normalize one embedding for easier curve comparison."""
    return (x - x.mean()) / (x.std() + 1e-8)


def visualize_speaker_embeddings(
    source_emb,
    target_emb,
    vc_emb,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Visualize source, target, and VC speaker embeddings.

    Parameters
    ----------
    source_emb, target_emb, vc_emb:
        Speaker embeddings as torch.Tensor or np.ndarray.
    save_path:
        Optional path to save the figure, e.g., "speaker_emb_viz.png".

    Returns
    -------
    dict
        Cosine similarity metrics.
    """
    source = _to_numpy(source_emb)
    target = _to_numpy(target_emb)
    vc = _to_numpy(vc_emb)

    metrics = {
        "source_target": _cosine(source, target),
        "vc_source": _cosine(vc, source),
        "vc_target": _cosine(vc, target),
    }
    metrics["target_gain"] = metrics["vc_target"] - metrics["vc_source"]

    print(f"sim Source ↔ Target: {metrics['source_target']:.4f}")
    print(f"sim VC     ↔ Source: {metrics['vc_source']:.4f}")
    print(f"sim VC     ↔ Target: {metrics['vc_target']:.4f}")
    print(f"Target gain:          {metrics['target_gain']:.4f}")

    names = ["Source", "Target", "VC"]
    embs = [source, target, vc]

    sim_mat = np.array([
        [_cosine(a, b) for b in embs]
        for a in embs
    ])

    source_z = _z_norm(source)
    target_z = _z_norm(target)
    vc_z = _z_norm(vc)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Similarity matrix
    ax = axes[0]
    im = ax.imshow(sim_mat, vmin=-1, vmax=1)
    ax.set_title("Cosine Similarity")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{sim_mat[i, j]:.3f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. Key similarity bars
    ax = axes[1]
    labels = ["Source–Target", "VC–Source", "VC–Target"]
    values = [
        metrics["source_target"],
        metrics["vc_source"],
        metrics["vc_target"],
    ]

    ax.bar(labels, values)
    ax.set_ylim(-1, 1)
    ax.axhline(0, linewidth=1)
    ax.set_title("Speaker Similarity")
    ax.set_ylabel("Cosine similarity")
    ax.tick_params(axis="x", rotation=20)

    for i, v in enumerate(values):
        ax.text(i, v + (0.03 if v >= 0 else -0.08), f"{v:.3f}", ha="center")

    # 3. Normalized embedding curves
    ax = axes[2]
    ax.plot(source_z, label="Source", alpha=0.85)
    ax.plot(target_z, label="Target", alpha=0.85)
    ax.plot(vc_z, label="VC", alpha=0.85)

    ax.set_title("Normalized Embeddings")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Z-normalized value")
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

    return metrics
