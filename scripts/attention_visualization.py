"""
Attention weight visualization for the Transformer-LSTM hybrid.

Extracts and visualizes the multi-head self-attention weights from
the Transformer encoder, showing which frames the model attends to
during fall vs ADL classification. This provides interpretability
evidence that the Transformer learns to focus on impact-moment frames.

Generates:
  - Per-sample attention heatmaps (fall vs ADL examples)
  - Averaged attention profile: fall class vs ADL class
  - Per-head attention breakdown
  - Publication-quality figures (PNG + PDF)

Usage:
    python scripts/attention_visualization.py
    python scripts/attention_visualization.py --fold 0
    python scripts/attention_visualization.py --model-dir models/tl_run4_d64_drop03

Reference: research/4.1 - Transformer attention provides interpretability
    advantage over LSTM black-box hidden states.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.features.extractor import FeatureExtractor
from src.models.transformer_lstm import FallDetectionTransformerLSTM
from src.training.dataset import build_dataset_from_processed
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def _load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(transformer_cfg: dict, dropout: float) -> FallDetectionTransformerLSTM:
    """Create a Transformer-LSTM model from config."""
    return FallDetectionTransformerLSTM(
        input_size=FeatureExtractor.NUM_FEATURES,
        d_model=transformer_cfg.get("d_model", 64),
        nhead=transformer_cfg.get("nhead", 4),
        num_encoder_layers=transformer_cfg.get("num_encoder_layers", 2),
        dim_feedforward=transformer_cfg.get("dim_feedforward", 128),
        lstm_hidden_size=transformer_cfg.get("lstm_hidden_size", 128),
        lstm_num_layers=transformer_cfg.get("lstm_num_layers", 1),
        dropout=dropout,
    )


def _extract_attention_single_pass(
    model: FallDetectionTransformerLSTM,
    x: torch.Tensor,
) -> List[np.ndarray]:
    """
    Run a single forward pass and manually extract attention weights.

    Instead of hooks, we manually run the Transformer encoder layers
    one at a time, calling self_attn directly with need_weights=True.

    Parameters
    ----------
    model : FallDetectionTransformerLSTM
        The model.
    x : torch.Tensor
        Input tensor of shape (1, T, input_size).

    Returns
    -------
    list of np.ndarray
        Per-layer attention weights, each shape (nhead, T, T).
    """
    # Step 1: Input projection + positional encoding
    h = model.input_projection(x)
    h = model.pos_encoder(h)

    layer_attentions = []

    # Step 2: Manually iterate through transformer encoder layers
    for layer in model.transformer_encoder.layers:
        # TransformerEncoderLayer._sa_block equivalent:
        # self_attn expects (query, key, value) all same for self-attention
        # We need to call self_attn directly with need_weights=True

        # Self-attention sub-block
        src = h
        # Apply self-attention with need_weights=True
        attn_output, attn_weights = layer.self_attn(
            src, src, src,
            need_weights=True,
            average_attn_weights=False,  # Get per-head weights
        )
        # attn_weights: (batch, nhead, T, T)

        layer_attentions.append(attn_weights[0].detach().cpu().numpy())

        # Complete the rest of the TransformerEncoderLayer forward:
        # residual + dropout + norm1
        h = layer.norm1(src + layer.dropout1(attn_output))

        # Feedforward sub-block
        ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
        h = layer.norm2(h + layer.dropout2(ff_output))

    return layer_attentions


def _validate_manual_forward(model: FallDetectionTransformerLSTM) -> None:
    """
    Validate that the manual forward pass produces the same output
    as the standard model.forward() call, guarding against API drift.
    """
    model.eval()
    x_test = torch.randn(1, 30, model.input_size)
    with torch.no_grad():
        standard_out = model(x_test)

        # Manual path: replicate the full forward
        h = model.input_projection(x_test)
        h = model.pos_encoder(h)
        for layer in model.transformer_encoder.layers:
            src = h
            attn_output, _ = layer.self_attn(
                src, src, src,
                need_weights=True, average_attn_weights=False,
            )
            h = layer.norm1(src + layer.dropout1(attn_output))
            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(h)))
            )
            h = layer.norm2(h + layer.dropout2(ff_output))
        _, (h_n, _) = model.lstm(h)
        hidden = h_n[-1]
        hidden = model.dropout(hidden)
        manual_out = model.fc(hidden)

    if not torch.allclose(standard_out, manual_out, atol=1e-5):
        raise RuntimeError(
            "Manual forward pass does not match standard model.forward()! "
            f"Max diff: {(standard_out - manual_out).abs().max().item():.6f}"
        )
    logger.info("Attention extraction validated: manual forward matches standard forward.")


def extract_attention_maps(
    model: FallDetectionTransformerLSTM,
    sequences: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Run inference and collect attention weights for all samples.

    Parameters
    ----------
    model : FallDetectionTransformerLSTM
        Model with attention hooks registered.
    sequences : np.ndarray
        Input sequences (N, T, F).
    labels : np.ndarray
        Ground truth labels (N,).
    max_samples : int
        Maximum samples to process per class.

    Returns
    -------
    dict
        Keys: 'fall_attn', 'adl_attn' — each shape (n_samples, n_layers, n_heads, T, T)
    """
    model.eval()

    # Separate fall and ADL indices
    fall_idx = np.where(labels == 1)[0]
    adl_idx = np.where(labels == 0)[0]

    # Subsample if too many
    if len(fall_idx) > max_samples:
        fall_idx = np.random.choice(fall_idx, max_samples, replace=False)
    if len(adl_idx) > max_samples:
        adl_idx = np.random.choice(adl_idx, max_samples, replace=False)

    def collect_attention(indices: np.ndarray) -> np.ndarray:
        """Collect attention weights for given sample indices."""
        all_attn = []
        for idx in indices:
            seq = torch.from_numpy(sequences[idx:idx+1].astype(np.float32))
            with torch.no_grad():
                layer_attns = _extract_attention_single_pass(model, seq)

            # layer_attns: list of (nhead, T, T), one per layer
            # Stack: (n_layers, nhead, T, T)
            all_attn.append(np.stack(layer_attns, axis=0))

        return np.array(all_attn)  # (n_samples, n_layers, nhead, T, T)

    fall_attn = collect_attention(fall_idx)
    adl_attn = collect_attention(adl_idx)

    logger.info(
        f"Collected attention: falls={fall_attn.shape}, ADLs={adl_attn.shape}"
    )

    return {
        "fall_attn": fall_attn,
        "adl_attn": adl_attn,
        "fall_idx": fall_idx,
        "adl_idx": adl_idx,
    }


def plot_attention_profiles(
    attn_data: Dict[str, np.ndarray],
    output_dir: Path,
    window_size: int = 30,
    fps: float = 15.0,
) -> None:
    """
    Generate publication-quality attention visualization figures.

    Creates:
    1. Average attention profile (which frames get attended to) for falls vs ADLs
    2. Per-head attention breakdown
    3. Example heatmaps for individual fall and ADL sequences
    4. Attention difference map (fall - ADL)

    Parameters
    ----------
    attn_data : dict
        Output from extract_attention_maps.
    output_dir : Path
        Directory to save figures.
    window_size : int
        Number of frames in each window.
    fps : float
        Frames per second for time axis labeling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fall_attn = attn_data["fall_attn"]  # (n_fall, n_layers, nhead, T, T)
    adl_attn = attn_data["adl_attn"]    # (n_adl, n_layers, nhead, T, T)

    n_layers = fall_attn.shape[1]
    n_heads = fall_attn.shape[2]
    T = fall_attn.shape[3]

    time_axis = np.arange(T) / fps  # seconds

    # =========================================================================
    # Figure 1: Average Attention Received per Frame (falls vs ADLs)
    # "Attention received" = column-sum of attention matrix (how much each
    # frame is attended TO by all other frames). This shows which temporal
    # positions the model considers most important.
    # =========================================================================
    # Average across samples, layers, heads, and query positions (rows)
    # For each frame j, sum attention[:,j] = how much frame j is attended to
    fall_received = fall_attn.mean(axis=(0, 1, 2))  # (T, T)
    fall_profile = fall_received.sum(axis=0)  # (T,) — total attention received
    fall_profile = fall_profile / fall_profile.sum()  # normalize to probability

    adl_received = adl_attn.mean(axis=(0, 1, 2))  # (T, T)
    adl_profile = adl_received.sum(axis=0)
    adl_profile = adl_profile / adl_profile.sum()

    # Uniform baseline
    uniform = np.ones(T) / T

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_axis, fall_profile, "r-", linewidth=2.0, label="Fall sequences")
    ax.plot(time_axis, adl_profile, "b-", linewidth=2.0, label="ADL sequences")
    ax.axhline(y=1.0/T, color="gray", linestyle="--", alpha=0.5, label="Uniform")
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Normalized Attention Received", fontsize=12)
    ax.set_title("Average Temporal Attention Profile: Fall vs ADL", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "attention_profile_fall_vs_adl.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "attention_profile_fall_vs_adl.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved attention_profile_fall_vs_adl")

    # =========================================================================
    # Figure 2: Attention Difference Map (Fall - ADL)
    # Shows where fall attention differs from ADL attention
    # =========================================================================
    fall_avg_map = fall_attn.mean(axis=(0, 1, 2))  # (T, T)
    adl_avg_map = adl_attn.mean(axis=(0, 1, 2))    # (T, T)
    diff_map = fall_avg_map - adl_avg_map

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    vmin_common = min(fall_avg_map.min(), adl_avg_map.min())
    vmax_common = max(fall_avg_map.max(), adl_avg_map.max())

    im0 = axes[0].imshow(fall_avg_map, aspect="auto", cmap="Reds",
                          vmin=vmin_common, vmax=vmax_common,
                          extent=[0, time_axis[-1], time_axis[-1], 0])
    axes[0].set_title("Fall Attention", fontsize=12)
    axes[0].set_xlabel("Key Frame (s)")
    axes[0].set_ylabel("Query Frame (s)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(adl_avg_map, aspect="auto", cmap="Blues",
                          vmin=vmin_common, vmax=vmax_common,
                          extent=[0, time_axis[-1], time_axis[-1], 0])
    axes[1].set_title("ADL Attention", fontsize=12)
    axes[1].set_xlabel("Key Frame (s)")
    axes[1].set_ylabel("Query Frame (s)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    vabs = max(abs(diff_map.min()), abs(diff_map.max()))
    im2 = axes[2].imshow(diff_map, aspect="auto", cmap="RdBu_r",
                          vmin=-vabs, vmax=vabs,
                          extent=[0, time_axis[-1], time_axis[-1], 0])
    axes[2].set_title("Difference (Fall \u2212 ADL)", fontsize=12)
    axes[2].set_xlabel("Key Frame (s)")
    axes[2].set_ylabel("Query Frame (s)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle("Average Self-Attention Maps (Averaged Over All Heads & Layers)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "attention_heatmaps_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "attention_heatmaps_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved attention_heatmaps_comparison")

    # =========================================================================
    # Figure 3: Per-Head Attention Profile (falls only)
    # Shows what each attention head specializes in
    # =========================================================================
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3.5 * n_heads, 3 * n_layers),
                              squeeze=False)
    fig.suptitle("Per-Head Attention Profiles (Fall Sequences)", fontsize=14, y=1.02)

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]
            # Average across fall samples: (n_fall, T, T) -> (T, T)
            head_attn = fall_attn[:, layer, head, :, :].mean(axis=0)
            # Attention received per frame
            profile = head_attn.sum(axis=0)
            profile = profile / profile.sum()

            ax.bar(time_axis, profile, width=1.0/fps * 0.8, color="firebrick", alpha=0.7)
            ax.axhline(y=1.0/T, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(f"Layer {layer+1}, Head {head+1}", fontsize=10)
            if layer == n_layers - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            if head == 0:
                ax.set_ylabel("Attn Weight", fontsize=9)
            ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "per_head_attention_falls.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "per_head_attention_falls.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per_head_attention_falls")

    # =========================================================================
    # Figure 4: Example individual heatmaps (3 falls + 3 ADLs)
    # =========================================================================
    n_examples = min(3, len(attn_data["fall_idx"]), len(attn_data["adl_idx"]))

    fig, axes = plt.subplots(2, n_examples, figsize=(4.5 * n_examples, 8), squeeze=False)
    fig.suptitle("Individual Sample Attention Maps (Avg Over Heads & Layers)", fontsize=14, y=1.02)

    for i in range(n_examples):
        # Fall example
        fall_map = fall_attn[i].mean(axis=(0, 1))  # (T, T)
        im = axes[0][i].imshow(fall_map, aspect="auto", cmap="Reds",
                                extent=[0, time_axis[-1], time_axis[-1], 0])
        axes[0][i].set_title(f"Fall Example {i+1}", fontsize=11)
        axes[0][i].set_xlabel("Key Frame (s)")
        if i == 0:
            axes[0][i].set_ylabel("Query Frame (s)")
        plt.colorbar(im, ax=axes[0][i], fraction=0.046)

        # ADL example
        adl_map = adl_attn[i].mean(axis=(0, 1))
        im = axes[1][i].imshow(adl_map, aspect="auto", cmap="Blues",
                                extent=[0, time_axis[-1], time_axis[-1], 0])
        axes[1][i].set_title(f"ADL Example {i+1}", fontsize=11)
        axes[1][i].set_xlabel("Key Frame (s)")
        if i == 0:
            axes[1][i].set_ylabel("Query Frame (s)")
        plt.colorbar(im, ax=axes[1][i], fraction=0.046)

    fig.tight_layout()
    fig.savefig(output_dir / "individual_attention_examples.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "individual_attention_examples.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved individual_attention_examples")

    # =========================================================================
    # Figure 5: Attention entropy (fall vs ADL)
    # Lower entropy = more focused attention; higher = more uniform
    # =========================================================================
    def attention_entropy(attn_maps: np.ndarray) -> np.ndarray:
        """Compute per-sample attention entropy averaged over layers/heads/queries."""
        # attn_maps: (n_samples, n_layers, n_heads, T, T)
        # For each query row, compute entropy of the attention distribution
        eps = 1e-10
        log_attn = np.log(attn_maps + eps)
        entropy = -(attn_maps * log_attn).sum(axis=-1)  # (n, layers, heads, T)
        # Average over layers, heads, query positions
        return entropy.mean(axis=(1, 2, 3))  # (n_samples,)

    fall_entropy = attention_entropy(fall_attn)
    adl_entropy = attention_entropy(adl_attn)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(
        min(fall_entropy.min(), adl_entropy.min()),
        max(fall_entropy.max(), adl_entropy.max()),
        30,
    )
    ax.hist(fall_entropy, bins=bins, alpha=0.6, color="red", label=f"Fall (n={len(fall_entropy)})", density=True)
    ax.hist(adl_entropy, bins=bins, alpha=0.6, color="blue", label=f"ADL (n={len(adl_entropy)})", density=True)
    ax.set_xlabel("Mean Attention Entropy", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Attention Focus: Fall vs ADL Sequences", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation with means
    ax.axvline(fall_entropy.mean(), color="red", linestyle="--", alpha=0.7)
    ax.axvline(adl_entropy.mean(), color="blue", linestyle="--", alpha=0.7)
    ax.text(0.02, 0.95, f"Fall mean: {fall_entropy.mean():.3f}\nADL mean: {adl_entropy.mean():.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_dir / "attention_entropy_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "attention_entropy_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved attention_entropy_distribution")

    # Save numeric summary
    summary = {
        "n_fall_samples": int(fall_attn.shape[0]),
        "n_adl_samples": int(adl_attn.shape[0]),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "window_size": int(T),
        "fall_entropy_mean": float(fall_entropy.mean()),
        "fall_entropy_std": float(fall_entropy.std()),
        "adl_entropy_mean": float(adl_entropy.mean()),
        "adl_entropy_std": float(adl_entropy.std()),
        "fall_attention_peak_frame": int(np.argmax(fall_profile)),
        "fall_attention_peak_time_s": float(np.argmax(fall_profile) / fps),
        "adl_attention_peak_frame": int(np.argmax(adl_profile)),
        "adl_attention_peak_time_s": float(np.argmax(adl_profile) / fps),
        "fall_profile": fall_profile.tolist(),
        "adl_profile": adl_profile.tolist(),
    }

    with open(output_dir / "attention_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved attention_summary.json")

    # Print key findings
    print("\n" + "=" * 60)
    print("ATTENTION VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"Fall samples: {fall_attn.shape[0]}, ADL samples: {adl_attn.shape[0]}")
    print(f"Architecture: {n_layers} layers, {n_heads} heads, {T} frames")
    print(f"\nAttention Entropy (lower = more focused):")
    print(f"  Fall: {fall_entropy.mean():.4f} +/- {fall_entropy.std():.4f}")
    print(f"  ADL:  {adl_entropy.mean():.4f} +/- {adl_entropy.std():.4f}")
    print(f"\nPeak attention frame:")
    print(f"  Fall: frame {np.argmax(fall_profile)} ({np.argmax(fall_profile)/fps:.2f}s)")
    print(f"  ADL:  frame {np.argmax(adl_profile)} ({np.argmax(adl_profile)/fps:.2f}s)")
    print(f"\nFigures saved to: {output_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention weight visualization")
    parser.add_argument(
        "--model-dir", type=str, default="models/tl_run4_d64_drop03",
        help="Path to the Transformer-LSTM model directory with fold subdirectories",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Specific fold to visualize (default: all folds)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/attention_visualization",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Max samples per class to process",
    )
    args = parser.parse_args()

    cfg = _load_config()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path(cfg["paths"]["data_processed"]) / "urfd"
    window_size = cfg["features"]["window_size"]
    stride = 2  # Same as training
    tl_cfg = cfg.get("transformer_lstm", {})
    dropout = cfg["model"]["dropout"]

    # Determine which folds to process
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = list(range(5))

    # Load subject splits
    metadata_path = processed_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    all_subjects = sorted(set(s["subject_id"] for s in metadata["sequences"]))
    logger.info(f"Subjects: {all_subjects}")

    # Collect attention from all folds (using test subjects)
    all_fall_attn = []
    all_adl_attn = []

    for fold_idx in folds:
        fold_dir = model_dir / f"fold_{fold_idx}"
        if not fold_dir.exists():
            logger.warning(f"Fold directory not found: {fold_dir}")
            continue

        # Load model
        model = _build_model(tl_cfg, dropout)
        ckpt_path = fold_dir / "best_model.pth"
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # Handle both raw state_dict and wrapped checkpoint formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
        model.eval()

        # Validate manual forward pass on first fold
        if fold_idx == folds[0]:
            _validate_manual_forward(model)

        # Load test data (held-out subject for this fold)
        test_subject = all_subjects[fold_idx]
        norm_mean = np.load(str(fold_dir / "norm_mean.npy"))
        norm_std = np.load(str(fold_dir / "norm_std.npy"))

        test_seqs, test_labels, _ = build_dataset_from_processed(
            processed_dir=processed_dir,
            subject_ids=[test_subject],
            window_size=window_size,
            stride=stride,
            norm_stats=(norm_mean, norm_std),
        )

        logger.info(
            f"Fold {fold_idx} (test={test_subject}): "
            f"{len(test_seqs)} windows, {(test_labels==1).sum()} falls, "
            f"{(test_labels==0).sum()} ADLs"
        )

        # Extract attention
        attn_data = extract_attention_maps(
            model, test_seqs, test_labels, max_samples=args.max_samples,
        )

        all_fall_attn.append(attn_data["fall_attn"])
        all_adl_attn.append(attn_data["adl_attn"])

    # Concatenate across folds
    combined_fall = np.concatenate(all_fall_attn, axis=0)
    combined_adl = np.concatenate(all_adl_attn, axis=0)

    combined_data = {
        "fall_attn": combined_fall,
        "adl_attn": combined_adl,
        "fall_idx": np.arange(len(combined_fall)),
        "adl_idx": np.arange(len(combined_adl)),
    }

    logger.info(
        f"Total: {combined_fall.shape[0]} fall windows, "
        f"{combined_adl.shape[0]} ADL windows across {len(folds)} folds"
    )

    # Generate all plots
    plot_attention_profiles(
        combined_data,
        output_dir=output_dir,
        window_size=window_size,
        fps=cfg["video"]["target_fps"],
    )


if __name__ == "__main__":
    main()
