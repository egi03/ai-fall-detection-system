"""
Inference benchmark for the fall detection pipeline.

Measures per-component and end-to-end latency on CPU to determine
real-time feasibility at 15 FPS target.

Components benchmarked:
  1. LSTM inference (single and batched)
  2. Feature extraction from keypoints
  3. Full pipeline estimate (sum of components)
  4. Model complexity (parameters, file size)

Usage:
    python scripts/inference_benchmark.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import FallDetectionLSTM
from src.features.extractor import FeatureExtractor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_WARMUP = 100
NUM_ITERATIONS = 1000
WINDOW_SIZE = 30
NUM_FEATURES = 15
NUM_KEYPOINTS_MP = 33  # MediaPipe BlazePose landmarks
BATCH_SIZES = [1, 4, 8, 16, 32]
TARGET_FPS = 15
MODEL_PATH = PROJECT_ROOT / "models" / "run5_stride2_aug" / "fold_0" / "best_model.pth"

# Model hyperparameters (from config/config.yaml)
MODEL_CONFIG = dict(
    input_size=15,
    hidden_size=128,
    num_layers=2,
    num_classes=2,
    dropout=0.5,
    bidirectional=True,
)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def estimate_flops_lstm(input_size: int, hidden_size: int, num_layers: int,
                        seq_len: int, bidirectional: bool) -> int:
    """
    Estimate FLOPs for LSTM forward pass.

    Each LSTM cell gate computation is roughly:
      4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    per timestep, per layer, per direction.
    Plus the linear output layer.
    """
    directions = 2 if bidirectional else 1
    flops_per_cell = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    # First layer takes input_size, subsequent layers take hidden_size * directions
    total_flops = 0
    for layer in range(num_layers):
        if layer == 0:
            layer_input = input_size
        else:
            layer_input = hidden_size * directions
        layer_flops = 4 * (layer_input * hidden_size + hidden_size * hidden_size + hidden_size)
        total_flops += layer_flops * seq_len * directions

    # Final linear layer: (hidden_size * directions) * num_classes
    total_flops += hidden_size * directions * 2  # num_classes=2
    return total_flops


def benchmark_lstm_inference(model: torch.nn.Module) -> dict:
    """Benchmark LSTM inference for various batch sizes."""
    model.eval()
    results = {}

    for batch_size in BATCH_SIZES:
        x = torch.randn(batch_size, WINDOW_SIZE, NUM_FEATURES)

        # Warmup
        with torch.no_grad():
            for _ in range(NUM_WARMUP):
                _ = model(x)

        # Timed iterations
        times = []
        with torch.no_grad():
            for _ in range(NUM_ITERATIONS):
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                times.append(t1 - t0)

        times_ms = np.array(times) * 1000.0
        results[batch_size] = {
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "median_ms": float(np.median(times_ms)),
            "p95_ms": float(np.percentile(times_ms, 95)),
            "p99_ms": float(np.percentile(times_ms, 99)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
        }

    return results


def benchmark_feature_extraction() -> dict:
    """Benchmark FeatureExtractor.extract() on synthetic keypoint data."""
    extractor = FeatureExtractor(keypoint_format="mediapipe", confidence_threshold=0.5)

    # Create realistic synthetic keypoints: (33, 4) with x, y, z, visibility
    rng = np.random.RandomState(42)
    keypoints = rng.rand(NUM_KEYPOINTS_MP, 4).astype(np.float32)
    keypoints[:, 3] = rng.uniform(0.6, 1.0, NUM_KEYPOINTS_MP).astype(np.float32)  # high visibility
    prev_keypoints = rng.rand(NUM_KEYPOINTS_MP, 4).astype(np.float32)
    prev_keypoints[:, 3] = rng.uniform(0.6, 1.0, NUM_KEYPOINTS_MP).astype(np.float32)

    # Warmup
    for _ in range(NUM_WARMUP):
        _ = extractor.extract(keypoints, prev_keypoints=prev_keypoints, dt=1.0 / 15.0, prev_velocity=0.1)

    # Timed iterations
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        _ = extractor.extract(keypoints, prev_keypoints=prev_keypoints, dt=1.0 / 15.0, prev_velocity=0.1)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_ms = np.array(times) * 1000.0
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }


def benchmark_feature_sequence() -> dict:
    """Benchmark extract_sequence() for a full 30-frame window."""
    extractor = FeatureExtractor(keypoint_format="mediapipe", confidence_threshold=0.5)

    rng = np.random.RandomState(42)
    keypoint_seq = rng.rand(WINDOW_SIZE, NUM_KEYPOINTS_MP, 4).astype(np.float32)
    keypoint_seq[:, :, 3] = rng.uniform(0.6, 1.0, (WINDOW_SIZE, NUM_KEYPOINTS_MP)).astype(np.float32)

    # Warmup
    for _ in range(NUM_WARMUP):
        _ = extractor.extract_sequence(keypoint_seq, dt=1.0 / 15.0)

    # Timed iterations (fewer since this is 30x slower)
    n_iter = 200
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = extractor.extract_sequence(keypoint_seq, dt=1.0 / 15.0)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_ms = np.array(times) * 1000.0
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "iterations": n_iter,
    }


def get_model_size(model_path: Path) -> dict:
    """Get model file size on disk."""
    size_bytes = os.path.getsize(model_path)
    return {
        "bytes": size_bytes,
        "kb": size_bytes / 1024,
        "mb": size_bytes / (1024 * 1024),
    }


def main() -> None:
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("FALL DETECTION INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"Device: CPU")
    print(f"Warmup iterations: {NUM_WARMUP}")
    print(f"Timed iterations: {NUM_ITERATIONS}")
    print(f"Window size: {WINDOW_SIZE} frames")
    print(f"Feature count: {NUM_FEATURES}")
    print(f"Target FPS: {TARGET_FPS}")
    print()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("[1/5] Loading model...")
    model = FallDetectionLSTM(**MODEL_CONFIG)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"  Model loaded from: {MODEL_PATH}")
    print()

    # ------------------------------------------------------------------
    # 2. Model complexity
    # ------------------------------------------------------------------
    print("[2/5] Model complexity...")
    params = count_parameters(model)
    flops = estimate_flops_lstm(
        input_size=MODEL_CONFIG["input_size"],
        hidden_size=MODEL_CONFIG["hidden_size"],
        num_layers=MODEL_CONFIG["num_layers"],
        seq_len=WINDOW_SIZE,
        bidirectional=MODEL_CONFIG["bidirectional"],
    )
    model_size = get_model_size(MODEL_PATH)

    print(f"  Total parameters:     {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Estimated FLOPs:      {flops:,} ({flops / 1e6:.2f} MFLOPs)")
    print(f"  Model file size:      {model_size['kb']:.1f} KB ({model_size['mb']:.3f} MB)")
    print()

    # ------------------------------------------------------------------
    # 3. LSTM inference benchmark
    # ------------------------------------------------------------------
    print("[3/5] Benchmarking LSTM inference...")
    lstm_results = benchmark_lstm_inference(model)

    print(f"  {'Batch':>5} | {'Mean (ms)':>10} | {'Std (ms)':>9} | {'Median':>9} | {'P95':>9} | {'P99':>9}")
    print(f"  {'-' * 5}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}")
    for bs, r in lstm_results.items():
        print(f"  {bs:>5} | {r['mean_ms']:>10.3f} | {r['std_ms']:>9.3f} | {r['median_ms']:>9.3f} | {r['p95_ms']:>9.3f} | {r['p99_ms']:>9.3f}")
    print()

    # ------------------------------------------------------------------
    # 4. Feature extraction benchmark
    # ------------------------------------------------------------------
    print("[4/5] Benchmarking feature extraction...")
    feat_results = benchmark_feature_extraction()
    seq_results = benchmark_feature_sequence()

    print(f"  Single frame extract():")
    print(f"    Mean:   {feat_results['mean_ms']:.4f} ms")
    print(f"    Std:    {feat_results['std_ms']:.4f} ms")
    print(f"    Median: {feat_results['median_ms']:.4f} ms")
    print(f"    P95:    {feat_results['p95_ms']:.4f} ms")
    print(f"    P99:    {feat_results['p99_ms']:.4f} ms")
    print()
    print(f"  Full sequence extract_sequence() ({WINDOW_SIZE} frames):")
    print(f"    Mean:   {seq_results['mean_ms']:.3f} ms")
    print(f"    Median: {seq_results['median_ms']:.3f} ms")
    print(f"    P95:    {seq_results['p95_ms']:.3f} ms")
    print()

    # ------------------------------------------------------------------
    # 5. Full pipeline estimate
    # ------------------------------------------------------------------
    print("[5/5] Full pipeline estimate...")
    print()

    # Per-frame budget at 15 FPS = 66.67 ms
    frame_budget_ms = 1000.0 / TARGET_FPS

    # Components that run every frame
    feature_per_frame_ms = feat_results['mean_ms']

    # LSTM runs once per window (every stride frames, typically stride=2)
    lstm_single_ms = lstm_results[1]['mean_ms']
    lstm_amortized_ms = lstm_single_ms / 2.0  # stride=2 means LSTM runs every 2 frames

    # Pose estimation is typically 20-40ms on CPU (MediaPipe)
    # YOLOv8n detection is typically 30-80ms on CPU
    # We report these as reference values, not measured here
    pose_est_typical_ms = 30.0  # MediaPipe Pose on CPU (literature value)
    detection_typical_ms = 50.0  # YOLOv8n on CPU (literature value)

    total_measured_ms = feature_per_frame_ms + lstm_amortized_ms
    total_with_estimates_ms = total_measured_ms + pose_est_typical_ms + detection_typical_ms

    max_fps_measured = 1000.0 / total_measured_ms if total_measured_ms > 0 else float("inf")
    max_fps_full = 1000.0 / total_with_estimates_ms if total_with_estimates_ms > 0 else float("inf")

    print("  Component Timing Summary:")
    print(f"  {'-' * 55}")
    print(f"  {'Component':<35} | {'Time (ms)':>10} | {'Source':>6}")
    print(f"  {'-' * 35}-+-{'-' * 10}-+-{'-' * 6}")
    print(f"  {'Feature extraction (per frame)':<35} | {feature_per_frame_ms:>10.3f} | {'meas.':>6}")
    print(f"  {'LSTM inference (batch=1)':<35} | {lstm_single_ms:>10.3f} | {'meas.':>6}")
    print(f"  {'LSTM amortized (stride=2)':<35} | {lstm_amortized_ms:>10.3f} | {'meas.':>6}")
    print(f"  {'Pose estimation (MediaPipe, typ.)':<35} | {pose_est_typical_ms:>10.1f} | {'est.':>6}")
    print(f"  {'Person detection (YOLOv8n, typ.)':<35} | {detection_typical_ms:>10.1f} | {'est.':>6}")
    print(f"  {'-' * 35}-+-{'-' * 10}-+-{'-' * 6}")
    print(f"  {'TOTAL (measured only)':<35} | {total_measured_ms:>10.3f} |")
    print(f"  {'TOTAL (with detection+pose est.)':<35} | {total_with_estimates_ms:>10.1f} |")
    print()
    print(f"  Frame budget at {TARGET_FPS} FPS: {frame_budget_ms:.2f} ms")
    print(f"  Max FPS (feature+LSTM only):  {max_fps_measured:.0f} FPS")
    print(f"  Max FPS (full pipeline est.):  {max_fps_full:.1f} FPS")
    print()

    feasible = total_with_estimates_ms < frame_budget_ms
    print(f"  Real-time at {TARGET_FPS} FPS: {'YES' if feasible else 'NO'} "
          f"({'%.1f' % (frame_budget_ms - total_with_estimates_ms)} ms margin)"
          if feasible else
          f"  Real-time at {TARGET_FPS} FPS: NO "
          f"({total_with_estimates_ms - frame_budget_ms:.1f} ms over budget)")
    print()

    # ------------------------------------------------------------------
    # Save results to markdown
    # ------------------------------------------------------------------
    output_dir = PROJECT_ROOT / "DAILY LOGS" / "15.3"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "INFERENCE_BENCHMARK.md"

    md_lines = []
    md_lines.append("# Inference Benchmark Results")
    md_lines.append("")
    md_lines.append(f"**Date:** 2026-03-16  ")
    md_lines.append(f"**Device:** CPU (Windows 11, no CUDA)  ")
    md_lines.append(f"**Model:** BiLSTM, Run5 fold_0 (best single model)  ")
    md_lines.append(f"**Warmup:** {NUM_WARMUP} iterations | **Timed:** {NUM_ITERATIONS} iterations  ")
    md_lines.append("")

    # Model complexity table
    md_lines.append("## Model Complexity")
    md_lines.append("")
    md_lines.append("| Metric | Value |")
    md_lines.append("|--------|-------|")
    md_lines.append(f"| Total parameters | {params['total']:,} |")
    md_lines.append(f"| Trainable parameters | {params['trainable']:,} |")
    md_lines.append(f"| Estimated FLOPs (single inference) | {flops:,} ({flops / 1e6:.2f} MFLOPs) |")
    md_lines.append(f"| Model file size | {model_size['kb']:.1f} KB ({model_size['mb']:.3f} MB) |")
    md_lines.append(f"| Architecture | BiLSTM, 2 layers, 128 hidden, dropout=0.5 |")
    md_lines.append(f"| Input shape | (batch, {WINDOW_SIZE}, {NUM_FEATURES}) |")
    md_lines.append("")

    # LSTM inference table
    md_lines.append("## LSTM Inference Latency")
    md_lines.append("")
    md_lines.append("| Batch Size | Mean (ms) | Std (ms) | Median (ms) | P95 (ms) | P99 (ms) |")
    md_lines.append("|------------|-----------|----------|-------------|----------|----------|")
    for bs, r in lstm_results.items():
        md_lines.append(
            f"| {bs} | {r['mean_ms']:.3f} | {r['std_ms']:.3f} | "
            f"{r['median_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} |"
        )
    md_lines.append("")

    # Feature extraction table
    md_lines.append("## Feature Extraction Latency")
    md_lines.append("")
    md_lines.append("| Operation | Mean (ms) | Std (ms) | Median (ms) | P95 (ms) | P99 (ms) |")
    md_lines.append("|-----------|-----------|----------|-------------|----------|----------|")
    md_lines.append(
        f"| Single frame extract() | {feat_results['mean_ms']:.4f} | "
        f"{feat_results['std_ms']:.4f} | {feat_results['median_ms']:.4f} | "
        f"{feat_results['p95_ms']:.4f} | {feat_results['p99_ms']:.4f} |"
    )
    md_lines.append(
        f"| Full sequence ({WINDOW_SIZE} frames) | {seq_results['mean_ms']:.3f} | "
        f"{seq_results['std_ms']:.3f} | {seq_results['median_ms']:.3f} | "
        f"{seq_results['p95_ms']:.3f} | -- |"
    )
    md_lines.append("")

    # Pipeline summary table
    md_lines.append("## Full Pipeline Timing Estimate")
    md_lines.append("")
    md_lines.append("| Component | Time (ms) | Source |")
    md_lines.append("|-----------|-----------|--------|")
    md_lines.append(f"| Person detection (YOLOv8n) | ~{detection_typical_ms:.0f} | literature estimate |")
    md_lines.append(f"| Pose estimation (MediaPipe) | ~{pose_est_typical_ms:.0f} | literature estimate |")
    md_lines.append(f"| Feature extraction (per frame) | {feature_per_frame_ms:.3f} | measured |")
    md_lines.append(f"| LSTM inference (batch=1) | {lstm_single_ms:.3f} | measured |")
    md_lines.append(f"| LSTM amortized (stride=2) | {lstm_amortized_ms:.3f} | measured |")
    md_lines.append(f"| **Total (measured components)** | **{total_measured_ms:.3f}** | |")
    md_lines.append(f"| **Total (full pipeline est.)** | **~{total_with_estimates_ms:.1f}** | |")
    md_lines.append("")

    # Feasibility analysis
    md_lines.append("## Real-Time Feasibility Analysis")
    md_lines.append("")
    md_lines.append(f"- **Target:** {TARGET_FPS} FPS = {frame_budget_ms:.2f} ms per frame")
    md_lines.append(f"- **Max FPS (feature + LSTM only):** ~{max_fps_measured:.0f} FPS")
    md_lines.append(f"- **Max FPS (full pipeline estimate):** ~{max_fps_full:.1f} FPS")
    md_lines.append("")

    if feasible:
        margin = frame_budget_ms - total_with_estimates_ms
        md_lines.append(
            f"The full pipeline is estimated to run within the {frame_budget_ms:.1f} ms "
            f"budget with ~{margin:.1f} ms margin. Real-time operation at {TARGET_FPS} FPS "
            f"is **feasible** on CPU."
        )
    else:
        overshoot = total_with_estimates_ms - frame_budget_ms
        md_lines.append(
            f"The full pipeline exceeds the {frame_budget_ms:.1f} ms budget by "
            f"~{overshoot:.1f} ms. Real-time operation at {TARGET_FPS} FPS "
            f"requires optimization (e.g., skip detection on some frames, "
            f"use lighter pose model, or reduce resolution)."
        )

    md_lines.append("")
    md_lines.append("Key observations:")
    md_lines.append("")
    md_lines.append(
        f"- The LSTM and feature extraction together consume only "
        f"~{total_measured_ms:.2f} ms per frame, which is negligible "
        f"(<{total_measured_ms / frame_budget_ms * 100:.1f}% of the frame budget)."
    )
    md_lines.append(
        f"- The computational bottleneck is person detection + pose estimation "
        f"(~{detection_typical_ms + pose_est_typical_ms:.0f} ms combined), "
        f"which accounts for >{(detection_typical_ms + pose_est_typical_ms) / total_with_estimates_ms * 100:.0f}% "
        f"of total pipeline time."
    )
    md_lines.append(
        f"- The model is extremely lightweight at {params['total']:,} parameters "
        f"and {model_size['kb']:.0f} KB, suitable for edge deployment."
    )
    md_lines.append("")

    # Paper paragraph
    md_lines.append("## Paper-Ready Paragraph")
    md_lines.append("")
    md_lines.append(
        f"The proposed fall detection system employs a lightweight BiLSTM classifier "
        f"with {params['total']:,} trainable parameters ({model_size['kb']:.0f} KB on disk), "
        f"requiring approximately {flops / 1e6:.1f} MFLOPs per inference. "
        f"On a consumer-grade CPU (Intel, Windows 11), single-sample LSTM inference "
        f"completes in {lstm_results[1]['mean_ms']:.2f} ms (median {lstm_results[1]['median_ms']:.2f} ms), "
        f"while per-frame feature extraction from 33 MediaPipe keypoints requires "
        f"{feat_results['mean_ms']:.3f} ms. "
        f"Together, the classification components (feature extraction + LSTM) "
        f"consume only ~{total_measured_ms:.1f} ms per frame, "
        f"representing less than {total_measured_ms / frame_budget_ms * 100:.1f}% of the "
        f"{frame_budget_ms:.1f} ms budget required for {TARGET_FPS} FPS real-time operation. "
        f"The computational bottleneck lies in the upstream perception modules "
        f"(YOLOv8n person detection and MediaPipe pose estimation), which typically "
        f"require ~{detection_typical_ms + pose_est_typical_ms:.0f} ms combined on CPU. "
        f"The full pipeline is estimated to achieve ~{max_fps_full:.0f} FPS, "
        f"{'meeting' if feasible else 'approaching'} the {TARGET_FPS} FPS target. "
        f"The model's minimal footprint ({model_size['kb']:.0f} KB, {params['total']:,} parameters) "
        f"makes it suitable for deployment on resource-constrained edge devices "
        f"such as Raspberry Pi or NVIDIA Jetson Nano."
    )
    md_lines.append("")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
