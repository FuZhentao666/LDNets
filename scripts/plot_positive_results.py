#!/usr/bin/env python3
"""Plot positive LDNets reproduction and sparse-encoder results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "runs"
    / "jepa"
    / "main_no_jepa_sparse_near_convergence_20260527"
    / "formal"
    / "summary.csv"
)
DEFAULT_STAGE_A = REPO_ROOT / "runs" / "case1c_alignment_stageA_20260528"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "results"

CASE_REFERENCES = {
    "1a": 8.576e-3,
    "1b": 2.440e-2,
    "1c": 2.039e-2,
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def number(value):
    if value in (None, ""):
        return None
    return float(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_case1_sensor_sweep(rows: list[dict[str, str]], output_dir: Path):
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        case = row["case"]
        if case not in {"1a", "1b"}:
            continue
        grouped[(case, number(row["sensor_ratio"]))].append(number(row["nrmse"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    palette = {"1a": "#2563eb", "1b": "#059669"}
    labels = {"1a": "Case 1a sparse", "1b": "Case 1b sparse"}

    for case in ("1a", "1b"):
        xs = sorted(sensor for (case_key, sensor), _ in grouped.items() if case_key == case)
        ys = [mean(grouped[(case, sensor)]) for sensor in xs]
        yerr = [sample_std(grouped[(case, sensor)]) for sensor in xs]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=palette[case],
            label=labels[case],
        )
        ax.axhline(
            CASE_REFERENCES[case],
            linestyle="--",
            linewidth=1.3,
            color=palette[case],
            alpha=0.45,
            label=f"Case {case} original ref",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sensor ratio")
    ax.set_ylabel("Test NRMSE")
    ax.set_title("Sparse encoder robustness on positive Case 1a/1b")
    ax.grid(True, which="both", linewidth=0.6, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "case1ab_sparse_sensor_sweep")


def plot_case1_all_cases(rows: list[dict[str, str]], output_dir: Path):
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["case"], number(row["sensor_ratio"]))].append(number(row["nrmse"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    palette = {"1a": "#2563eb", "1b": "#059669", "1c": "#dc2626"}
    for case in ("1a", "1b", "1c"):
        xs = sorted(sensor for (case_key, sensor), _ in grouped.items() if case_key == case)
        ys = [mean(grouped[(case, sensor)]) for sensor in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=palette[case], label=f"Case {case}")
        ax.axhline(
            CASE_REFERENCES[case],
            linestyle="--",
            linewidth=1.1,
            color=palette[case],
            alpha=0.35,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sensor ratio")
    ax.set_ylabel("Mean test NRMSE")
    ax.set_title("Near-convergence sparse sweep across Case 1")
    ax.grid(True, which="both", linewidth=0.6, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "case1_sparse_sensor_sweep_all_cases")


def stage_a_rows(stage_a_dir: Path) -> list[dict]:
    specs = [
        (
            "original_adam4000_bfgs150_seed0",
            "Original",
            "full",
            "Adam4000+BFGS150",
            "TestCase_1c_metrics.json",
        ),
        (
            "original_adam200_bfgs1800_seed0_gpu",
            "Original",
            "full",
            "Adam200+BFGS1800",
            "TestCase_1c_metrics.json",
        ),
        (
            "sparse_sr1p0_adam4000_bfgs150_seed0_gpu",
            "Sparse",
            "1.0",
            "Adam4000+BFGS150",
            "metrics.json",
        ),
        (
            "sparse_sr0p2_adam4000_bfgs150_seed0_gpu",
            "Sparse",
            "0.2",
            "Adam4000+BFGS150",
            "metrics.json",
        ),
        (
            "sparse_sr1p0_adam200_bfgs1800_seed0_gpu",
            "Sparse",
            "1.0",
            "Adam200+BFGS1800",
            "metrics.json",
        ),
        (
            "sparse_sr0p2_adam200_bfgs1800_seed0_gpu",
            "Sparse",
            "0.2",
            "Adam200+BFGS1800",
            "metrics.json",
        ),
    ]
    rows = []
    for directory, method, sensor_ratio, budget, filename in specs:
        payload = load_json(stage_a_dir / directory / filename)
        rows.append(
            {
                "method": method,
                "sensor_ratio": sensor_ratio,
                "budget": budget,
                "nrmse": float(payload["nrmse"]),
                "pearson": float(payload["pearson_dissimilarity"]),
                "elapsed_hours": float(payload["elapsed_seconds"]) / 3600.0,
            }
        )
    return rows


def plot_case1c_alignment(rows: list[dict], output_dir: Path):
    labels = [
        "Orig\n4000+150",
        "Orig\n200+1800",
        "Sparse 1.0\n4000+150",
        "Sparse 0.2\n4000+150",
        "Sparse 1.0\n200+1800",
        "Sparse 0.2\n200+1800",
    ]
    values = [row["nrmse"] for row in rows]
    colors = ["#64748b", "#334155", "#f97316", "#fb923c", "#059669", "#10b981"]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(range(len(values)), values, color=colors, width=0.72)
    ax.axhline(
        CASE_REFERENCES["1c"],
        color="#111827",
        linestyle="--",
        linewidth=1.3,
        label="Original Case 1c ref",
    )
    for index, value in enumerate(values):
        ax.text(index, value * 1.015, f"{value:.3e}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Test NRMSE")
    ax.set_title("Case 1c alignment: long BFGS changes the sparse conclusion")
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "case1c_stageA_alignment")


def plot_case1c_time(rows: list[dict], output_dir: Path):
    labels = [
        "Orig\n4000+150",
        "Orig\n200+1800",
        "Sparse 1.0\n4000+150",
        "Sparse 0.2\n4000+150",
        "Sparse 1.0\n200+1800",
        "Sparse 0.2\n200+1800",
    ]
    values = [row["elapsed_hours"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(range(len(values)), values, color="#475569", width=0.72)
    for index, value in enumerate(values):
        ax.text(index, value + 0.03, f"{value:.2f}h", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Elapsed hours")
    ax.set_title("Case 1c Stage A runtime cost")
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "case1c_stageA_runtime")


def plot_gpu_multiprocess_smoke(output_dir: Path):
    labels = ["Formal run", "Smoke run", "Other/display"]
    values = [1768, 1000, 3226 - 1768 - 1000]
    colors = ["#2563eb", "#f97316", "#94a3b8"]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bottom = 0
    for label, value, color in zip(labels, values, colors):
        ax.bar(["GPU0 during smoke"], [value], bottom=[bottom], label=label, color=color)
        bottom += value
    ax.axhline(24564, linestyle="--", linewidth=1.2, color="#111827", label="RTX 4090 memory")
    ax.set_ylabel("Memory MiB")
    ax.set_title("One-GPU multi-process smoke: memory headroom")
    ax.set_ylim(0, 26000)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "gpu0_multiprocess_memory_smoke")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--stage-a-dir", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main():
    args = build_parser().parse_args()
    rows = load_csv(args.summary_csv)
    stage_rows = stage_a_rows(args.stage_a_dir)
    plot_case1_sensor_sweep(rows, args.output_dir)
    plot_case1_all_cases(rows, args.output_dir)
    plot_case1c_alignment(stage_rows, args.output_dir)
    plot_case1c_time(stage_rows, args.output_dir)
    plot_gpu_multiprocess_smoke(args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
