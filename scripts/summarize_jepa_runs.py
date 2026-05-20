#!/usr/bin/env python3
"""Summarize JEPA-LDNet run directories into reproducible tables."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


METRIC_FIELDS = [
    "nrmse",
    "pearson_dissimilarity",
    "sensor_nrmse",
    "reconstruction_mse_valid",
    "jepa_feature_mse_valid",
    "latent_smoothness_valid",
    "loss_train_last",
    "loss_valid_last",
    "elapsed_seconds",
    "horizon_nrmse_first",
    "horizon_nrmse_final",
    "horizon_nrmse_max",
    "parameter_count",
]


ROW_FIELDS = [
    "run_dir",
    "run_status",
    "case",
    "seed",
    "sensor_ratio",
    "sensor_count",
    "adam_epochs",
    "bfgs_epochs",
    "learning_rate",
    "batch_samples",
    "warmup_epochs",
    "jepa_ramp_epochs",
    "lambda_jepa",
    "lambda_smooth",
    "ema_decay",
    "context_steps",
    "target_mode",
    "mask_ratio",
    "prediction_horizon",
    "target_points",
    "target_count",
    "time_count",
    "target_time_strategy",
    "embedding_dim",
    "encoder_width",
    "predictor_width",
    "git_commit",
    "dirty_worktree",
    "cuda_visible_devices",
    "command_hash",
] + METRIC_FIELDS


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def get_arg(config, name, default=None):
    return config.get("args", {}).get(name, default)


def number_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sequence_metric(values, reducer):
    if not isinstance(values, list):
        return None
    numbers = [number_or_none(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    if reducer == "first":
        return numbers[0]
    if reducer == "final":
        return numbers[-1]
    if reducer == "max":
        return max(numbers)
    raise ValueError(f"Unknown reducer: {reducer}")


def row_from_metrics(metrics_path):
    metrics = load_json(metrics_path)
    config_path = metrics_path.with_name("config.json")
    config = metrics.get("config") or (load_json(config_path) if config_path.exists() else {})
    sensor_indices = config.get("sensor_indices", [])
    run_status = metrics.get("run_status") or config.get("run_status") or "completed"
    target_mode = get_arg(
        config,
        "target_mode",
        config.get("target_selection", {}).get("mode") or "points",
    )
    mask_ratio = get_arg(config, "mask_ratio", config.get("target_selection", {}).get("mask_ratio"))
    if target_mode == "points":
        mask_ratio = None
    target_time_strategy = get_arg(
        config,
        "target_time_strategy",
        config.get("target_selection", {}).get("time_strategy") or "next",
    )
    prediction_horizon = get_arg(
        config,
        "prediction_horizon",
        config.get("target_selection", {}).get("prediction_horizon") or 1,
    )
    target_points = get_arg(
        config,
        "target_points",
        config.get("target_selection", {}).get("target_points"),
    )
    row = {
        "run_dir": str(metrics_path.parent),
        "run_status": run_status,
        "case": metrics.get("case") or config.get("case"),
        "seed": get_arg(config, "seed"),
        "sensor_ratio": get_arg(config, "sensor_ratio"),
        "sensor_count": len(sensor_indices) if sensor_indices else get_arg(config, "sensor_count"),
        "adam_epochs": get_arg(config, "adam_epochs"),
        "bfgs_epochs": get_arg(config, "bfgs_epochs"),
        "learning_rate": get_arg(config, "learning_rate"),
        "batch_samples": get_arg(config, "batch_samples"),
        "warmup_epochs": config.get("loss", {}).get("warmup_epochs"),
        "jepa_ramp_epochs": config.get("loss", {}).get("jepa_ramp_epochs"),
        "lambda_jepa": config.get("loss", {}).get("lambda_jepa"),
        "lambda_smooth": config.get("loss", {}).get("lambda_smooth"),
        "ema_decay": config.get("loss", {}).get("ema_decay"),
        "context_steps": get_arg(
            config,
            "context_steps",
            config.get("target_selection", {}).get("context_steps"),
        ),
        "target_mode": target_mode,
        "mask_ratio": mask_ratio,
        "prediction_horizon": prediction_horizon,
        "target_points": target_points,
        "target_count": config.get("target_selection", {}).get("target_count"),
        "time_count": config.get("target_selection", {}).get("time_count"),
        "target_time_strategy": target_time_strategy,
        "embedding_dim": get_arg(config, "embedding_dim"),
        "encoder_width": get_arg(config, "encoder_width"),
        "predictor_width": get_arg(config, "predictor_width"),
        "git_commit": config.get("git_commit"),
        "dirty_worktree": bool(config.get("git_status_short")),
        "cuda_visible_devices": config.get("cuda_visible_devices"),
        "command_hash": config.get("command_hash"),
    }
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field)
    row["horizon_nrmse_first"] = sequence_metric(metrics.get("horizon_nrmse"), "first")
    row["horizon_nrmse_final"] = sequence_metric(metrics.get("horizon_nrmse"), "final")
    row["horizon_nrmse_max"] = sequence_metric(metrics.get("horizon_nrmse"), "max")
    row["parameter_count"] = metrics.get("parameter_count")
    return row


def find_metrics(paths, include_run_regex=None, exclude_run_regex=None):
    include_pattern = re.compile(include_run_regex) if include_run_regex else None
    exclude_pattern = re.compile(exclude_run_regex) if exclude_run_regex else None
    metrics_paths = []
    for root in paths:
        root = Path(root)
        if root.is_file() and root.name == "metrics.json":
            metrics_paths.append(root)
        elif root.exists():
            metrics_paths.extend(root.rglob("metrics.json"))

    selected_paths = []
    for metrics_path in sorted(set(metrics_paths)):
        run_dir = str(metrics_path.parent)
        if include_pattern and not include_pattern.search(run_dir):
            continue
        if exclude_pattern and exclude_pattern.search(run_dir):
            continue
        selected_paths.append(metrics_path)
    return selected_paths


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_FIELDS})


def markdown_table(rows, fields):
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def aggregate_rows(rows, group_fields):
    groups = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        groups.setdefault(key, []).append(row)

    aggregate = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        entry = {field: value for field, value in zip(group_fields, key)}
        entry["count"] = len(group)
        for metric in ["nrmse", "pearson_dissimilarity", "sensor_nrmse", "elapsed_seconds"]:
            values = [number_or_none(row.get(metric)) for row in group]
            values = [value for value in values if value is not None]
            if values:
                entry[f"{metric}_mean"] = statistics.fmean(values)
                entry[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(entry)
    return aggregate


def write_markdown(rows, aggregate, path, group_fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    detail_fields = [
        "run_dir",
        "run_status",
        "case",
        "seed",
        "sensor_ratio",
        "sensor_count",
        "batch_samples",
        "lambda_jepa",
        "lambda_smooth",
        "ema_decay",
        "context_steps",
        "target_mode",
        "mask_ratio",
        "prediction_horizon",
        "target_points",
        "target_count",
        "time_count",
        "target_time_strategy",
        "nrmse",
        "pearson_dissimilarity",
        "sensor_nrmse",
        "horizon_nrmse_final",
        "horizon_nrmse_max",
        "elapsed_seconds",
    ]
    aggregate_fields = group_fields + [
        "count",
        "nrmse_mean",
        "nrmse_std",
        "pearson_dissimilarity_mean",
        "pearson_dissimilarity_std",
        "sensor_nrmse_mean",
        "sensor_nrmse_std",
        "elapsed_seconds_mean",
    ]
    content = [
        "# JEPA-LDNet Run Summary",
        "",
        "## Aggregate",
        "",
        markdown_table(aggregate, aggregate_fields),
        "",
        "## Runs",
        "",
        markdown_table(rows, detail_fields),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--include-run-regex",
        help="Only include runs whose directory path matches this regular expression.",
    )
    parser.add_argument(
        "--exclude-run-regex",
        help="Exclude runs whose directory path matches this regular expression.",
    )
    parser.add_argument(
        "--aggregate-by",
        default=(
            "case,sensor_ratio,lambda_jepa,lambda_smooth,ema_decay,"
            "target_mode,mask_ratio,prediction_horizon,target_points,target_time_strategy"
        ),
        help="Comma-separated row fields used for aggregate mean/std tables.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    rows = [
        row_from_metrics(path)
        for path in find_metrics(
            args.paths,
            include_run_regex=args.include_run_regex,
            exclude_run_regex=args.exclude_run_regex,
        )
    ]
    group_fields = [field.strip() for field in args.aggregate_by.split(",") if field.strip()]
    aggregate = aggregate_rows(rows, group_fields)

    if args.output_csv:
        write_csv(rows, args.output_csv)
    if args.output_md:
        write_markdown(rows, aggregate, args.output_md, group_fields)
    if not args.output_csv and not args.output_md:
        print(markdown_table(rows, ROW_FIELDS))


if __name__ == "__main__":
    main()
