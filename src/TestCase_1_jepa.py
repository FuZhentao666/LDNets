#!/usr/bin/env python3
"""JEPA-LDNet runner for ADR TestCase 1a/1b/1c."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import losses
import metrics
import models
import optimization
import utils
from TestCase_1 import CASE_CONFIGS, configure_gpus, dataset_chunks, load_case_data


tf.keras.backend.set_floatx("float64")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


class TrainingHistory:
    def __init__(self):
        self.iterations_history = []
        self.loss_train_history = []
        self.loss_valid_history = []

    def append(self, iteration, loss_train, loss_valid):
        self.iterations_history.append(int(iteration))
        self.loss_train_history.append(float(loss_train))
        self.loss_valid_history.append(float(loss_valid))
        print(
            "epoch% 5d   -   training loss: %1.3e   -   validation loss %1.3e"
            % (iteration, loss_train, loss_valid)
        )


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def current_git_status():
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def command_text():
    return " ".join([sys.executable] + sys.argv)


def command_hash(command):
    return hashlib.sha256(command.encode("utf-8")).hexdigest()[:16]


def gpu_memory_info():
    try:
        return tf.config.experimental.get_memory_info("GPU:0")
    except Exception:
        return None


def configure_and_list_gpus():
    configure_gpus()
    return [gpu.name for gpu in tf.config.list_physical_devices("GPU")]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["1a", "1b", "1c"], default="1a")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "jepa" / "case1a" / "smoke_seed0",
    )
    parser.add_argument("--adam-epochs", type=int, default=None)
    parser.add_argument("--bfgs-epochs", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-samples", type=int, default=25)
    parser.add_argument("--sensor-ratio", type=float, default=1.0)
    parser.add_argument("--sensor-count", type=int, default=None)
    parser.add_argument("--context-steps", type=int, default=1)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--target-points", type=int, default=32)
    parser.add_argument(
        "--target-mode",
        choices=["points", "masked-points", "future-patches"],
        default="points",
        help="How to select JEPA target points; 'points' preserves the original behavior.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.5,
        help="Maximum fraction of candidate hidden points used by masked target modes.",
    )
    parser.add_argument(
        "--target-time-strategy",
        choices=["next", "random-future", "horizon"],
        default="next",
        help="How to select future JEPA target times.",
    )
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--encoder-width", type=int, default=32)
    parser.add_argument("--predictor-width", type=int, default=32)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-jepa", type=float, default=0.1)
    parser.add_argument("--lambda-smooth", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-epochs",
        "--jepa-warmup-epochs",
        dest="jepa_warmup_epochs",
        type=int,
        default=10,
    )
    parser.add_argument("--jepa-ramp-epochs", type=int, default=20)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def condition_dim(config):
    problem = config["problem"]
    return len(problem["input_parameters"]) + len(problem["input_signals"])


def make_condition_sequence(dataset, config):
    batch = dataset["num_samples"]
    num_times = dataset["num_times"]
    dim = condition_dim(config)
    if dim == 0:
        return tf.zeros((batch, num_times, 0), dtype=tf.float64)
    if config["driver"] == "parameters":
        return tf.broadcast_to(
            tf.expand_dims(dataset["inp_parameters"], axis=1),
            (batch, num_times, dim),
        )
    return dataset["inp_signals"]


def make_condition_summary(dataset, config, context_steps):
    dim = condition_dim(config)
    if dim == 0:
        return tf.zeros((dataset["num_samples"], 0), dtype=tf.float64)
    if config["driver"] == "parameters":
        return dataset["inp_parameters"]
    return tf.reduce_mean(dataset["inp_signals"][:, :context_steps, :], axis=1)


def choose_indices(num_points, args, rng):
    if args.sensor_count is None:
        sensor_count = int(round(num_points * args.sensor_ratio))
    else:
        sensor_count = args.sensor_count
    sensor_count = max(1, min(num_points, sensor_count))
    sensor_indices = np.sort(rng.choice(num_points, sensor_count, replace=False))

    all_indices = np.arange(num_points)
    if args.target_mode in {"masked-points", "future-patches"}:
        candidate_indices = np.setdiff1d(all_indices, sensor_indices, assume_unique=True)
        if candidate_indices.size == 0:
            candidate_indices = all_indices
        mask_count = int(round(candidate_indices.size * args.mask_ratio))
        target_count = max(1, min(candidate_indices.size, args.target_points, mask_count))
    else:
        candidate_indices = all_indices
        target_count = max(1, min(num_points, args.target_points))

    if args.target_mode == "future-patches" and candidate_indices.size > target_count:
        start = rng.integers(0, candidate_indices.size - target_count + 1)
        target_indices = candidate_indices[start : start + target_count]
    else:
        target_indices = rng.choice(candidate_indices, target_count, replace=False)
    target_indices = np.sort(target_indices)
    return sensor_indices.astype(np.int32), target_indices.astype(np.int32)


def target_time_indices(num_times, args, rng):
    start = min(max(1, args.context_steps), num_times - 1)
    count = max(1, min(args.prediction_horizon, num_times - start))
    if args.target_time_strategy == "random-future":
        candidates = np.arange(start, num_times, dtype=np.int32)
        return np.sort(rng.choice(candidates, count, replace=False)).astype(np.int32)
    if args.target_time_strategy == "horizon":
        return np.unique(
            np.linspace(start, num_times - 1, count, dtype=np.int32)
        ).astype(np.int32)
    return np.arange(start, start + count, dtype=np.int32)


def gather_context_features(dataset, sensor_indices, context_steps):
    points = tf.convert_to_tensor(dataset["points_full"], dtype=tf.float64)
    fields = dataset["out_fields"]
    context_steps = min(context_steps, dataset["num_times"])
    points = tf.gather(points[:, :context_steps, :, :], sensor_indices, axis=2)
    fields = tf.gather(fields[:, :context_steps, :, :], sensor_indices, axis=2)
    times = tf.convert_to_tensor(dataset["times"][:context_steps], dtype=tf.float64)
    times = tf.broadcast_to(
        tf.reshape(times, (1, context_steps, 1, 1)),
        (dataset["num_samples"], context_steps, tf.shape(points)[2], 1),
    )
    features = tf.concat([points, times, fields], axis=-1)
    return tf.reshape(
        features,
        (dataset["num_samples"], context_steps * tf.shape(points)[2], tf.shape(features)[-1]),
    )


def gather_target_features(dataset, point_indices, time_indices):
    points = tf.convert_to_tensor(dataset["points_full"], dtype=tf.float64)
    fields = dataset["out_fields"]
    points = tf.gather(tf.gather(points, time_indices, axis=1), point_indices, axis=2)
    fields = tf.gather(tf.gather(fields, time_indices, axis=1), point_indices, axis=2)
    num_targets = len(time_indices)
    times_np = np.asarray(dataset["times"])[time_indices]
    times = tf.convert_to_tensor(times_np, dtype=tf.float64)
    times = tf.broadcast_to(
        tf.reshape(times, (1, num_targets, 1, 1)),
        (dataset["num_samples"], num_targets, tf.shape(points)[2], 1),
    )
    return tf.concat([points, times, fields], axis=-1)


def forward_chunk(model, chunk, config, args, sensor_indices, target_indices, time_indices):
    context_features = gather_context_features(chunk, sensor_indices, args.context_steps)
    target_features = gather_target_features(chunk, target_indices, time_indices)
    condition_summary = make_condition_summary(chunk, config, args.context_steps)
    conditions = make_condition_sequence(chunk, config)
    z0, context_embedding = model.encode_context(context_features, condition_summary)
    states = model.rollout(
        z0,
        conditions,
        tf.constant(config["dt"] / config["normalization"]["time"]["time_constant"], tf.float64),
    )
    prediction = model.decode(
        states, tf.convert_to_tensor(chunk["points_full"], dtype=tf.float64)
    )
    states_at_targets = tf.gather(states, time_indices, axis=1)
    target_times = tf.convert_to_tensor(np.asarray(chunk["times"])[time_indices], tf.float64)
    h_pred = model.predict_targets(
        states_at_targets, target_times, condition_summary, context_embedding
    )
    h_target = model.encode_targets(target_features, condition_summary)
    return prediction, states, h_pred, h_target


def loss_components_on_chunks(
    model,
    chunks,
    config,
    args,
    sensor_indices,
    target_indices,
    time_indices,
):
    rec_sum = tf.constant(0.0, dtype=tf.float64)
    rec_count = tf.constant(0.0, dtype=tf.float64)
    jepa_terms = []
    smooth_terms = []
    for chunk in chunks:
        prediction, states, h_pred, h_target = forward_chunk(
            model, chunk, config, args, sensor_indices, target_indices, time_indices
        )
        error = prediction - chunk["out_fields"]
        rec_sum += tf.reduce_sum(tf.square(error))
        rec_count += tf.cast(tf.size(error), tf.float64)
        jepa_terms.append(losses.jepa_loss(h_pred, h_target))
        smooth_terms.append(losses.latent_smoothness_loss(states))

    return {
        "reconstruction": rec_sum / rec_count,
        "jepa": tf.add_n(jepa_terms) / len(jepa_terms),
        "smoothness": tf.add_n(smooth_terms) / len(smooth_terms),
    }


def loss_on_chunks(
    model,
    chunks,
    config,
    args,
    sensor_indices,
    target_indices,
    time_indices,
    lambda_jepa_value,
):
    components = loss_components_on_chunks(
        model,
        chunks,
        config,
        args,
        sensor_indices,
        target_indices,
        time_indices,
    )
    loss_value = args.lambda_rec * components["reconstruction"]
    if args.lambda_jepa > 0:
        loss_value += lambda_jepa_value * components["jepa"]
    if args.lambda_smooth > 0:
        loss_value += args.lambda_smooth * components["smoothness"]
    if config["alpha_reg"] is not None:
        loss_value += config["alpha_reg"] * losses.weight_l2(
            [model.context_encoder, model.transition, model.decoder, model.predictor]
        )
    return loss_value


def prediction_on_chunks(model, chunks, config, args, sensor_indices, target_indices, time_indices):
    predictions = []
    for chunk in chunks:
        prediction, _, _, _ = forward_chunk(
            model, chunk, config, args, sensor_indices, target_indices, time_indices
        )
        predictions.append(prediction)
    return tf.concat(predictions, axis=0)


def build_model(config, args):
    problem = config["problem"]
    feature_dim = problem["space"]["dimension"] + 1 + len(problem["output_fields"])
    return models.JEPALDNet(
        feature_dim=feature_dim,
        condition_dim=condition_dim(config),
        latent_dim=config["num_latent_states"],
        embedding_dim=args.embedding_dim,
        space_dim=problem["space"]["dimension"],
        output_dim=len(problem["output_fields"]),
        dynamics_width=config["dynamics_width"],
        reconstruction_width=config["reconstruction_width"],
        encoder_width=args.encoder_width,
        predictor_width=args.predictor_width,
    )


def initialize_model(model, chunk, config, args, sensor_indices, target_indices, time_indices):
    forward_chunk(model, chunk, config, args, sensor_indices, target_indices, time_indices)
    model.sync_target_encoder()


def save_loss_plot(history, adam_epochs, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(history.iterations_history, history.loss_train_history, "o-", label="training loss")
    ax.plot(history.iterations_history, history.loss_valid_history, "o-", label="validation loss")
    if adam_epochs > 0:
        ax.axvline(adam_epochs)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_adam(model, loss_train, loss_valid, args):
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    history = TrainingHistory()
    lambda_jepa = tf.Variable(0.0, dtype=tf.float64, trainable=False)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss_value = loss_train(lambda_jepa)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        model.update_target_encoder(args.ema_decay)
        return loss_value

    history.append(0, loss_train(lambda_jepa).numpy(), loss_valid(lambda_jepa).numpy())
    for epoch in range(1, args.adam_epochs + 1):
        if epoch <= args.jepa_warmup_epochs:
            lambda_value = 0.0
        elif args.jepa_ramp_epochs <= 0:
            lambda_value = args.lambda_jepa
        else:
            lambda_value = args.lambda_jepa * min(
                1.0, (epoch - args.jepa_warmup_epochs) / args.jepa_ramp_epochs
            )
        lambda_jepa.assign(lambda_value)
        train_step()
        if epoch % 10 == 0 or epoch == args.adam_epochs:
            history.append(
                epoch,
                loss_train(lambda_jepa).numpy(),
                loss_valid(tf.constant(args.lambda_jepa, dtype=tf.float64)).numpy(),
            )
    return history, lambda_jepa


def append_bfgs_history(history, opt, start_iteration):
    for iteration, train_loss, valid_loss in zip(
        opt.iterations_history,
        opt.loss_train_history,
        opt.loss_valid_history,
    ):
        if iteration == 0:
            continue
        history.append(
            start_iteration + int(iteration),
            float(train_loss.numpy()),
            float(valid_loss.numpy()),
        )


def run(args):
    start_time = time.time()
    start_time_text = time.strftime("%Y-%m-%d %H:%M:%S %z")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_names = configure_and_list_gpus()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    config = CASE_CONFIGS[args.case]
    if args.adam_epochs is None:
        args.adam_epochs = config["adam_epochs"]

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset_train, dataset_valid, dataset_tests = load_case_data(config)
    sensor_indices, target_indices = choose_indices(dataset_train["num_points"], args, rng)
    time_indices = target_time_indices(dataset_train["num_times"], args, rng)

    train_chunks = dataset_chunks(dataset_train, args.batch_samples)
    valid_chunks = dataset_chunks(dataset_valid, args.batch_samples)
    test_chunks = dataset_chunks(dataset_tests, args.batch_samples)

    model = build_model(config, args)
    initialize_model(
        model,
        train_chunks[0],
        config,
        args,
        sensor_indices,
        target_indices,
        time_indices,
    )
    print("Trainable parameters:", metrics.parameter_count(model))

    def loss_train(lambda_jepa_value):
        return loss_on_chunks(
            model,
            train_chunks,
            config,
            args,
            sensor_indices,
            target_indices,
            time_indices,
            lambda_jepa_value,
        )

    def loss_valid(lambda_jepa_value):
        return loss_on_chunks(
            model,
            valid_chunks,
            config,
            args,
            sensor_indices,
            target_indices,
            time_indices,
            lambda_jepa_value,
        )

    history, lambda_jepa = train_adam(model, loss_train, loss_valid, args)

    if args.bfgs_epochs > 0:
        print("training (BFGS, fixed EMA target encoder)...")

        def bfgs_train():
            return loss_train(tf.constant(args.lambda_jepa, dtype=tf.float64))

        def bfgs_valid():
            return loss_valid(tf.constant(args.lambda_jepa, dtype=tf.float64))

        opt = optimization.OptimizationProblem(
            model.trainable_variables, bfgs_train, bfgs_valid
        )
        opt.optimize_BFGS(args.bfgs_epochs)
        append_bfgs_history(history, opt, args.adam_epochs)

    out_fields = prediction_on_chunks(
        model, test_chunks, config, args, sensor_indices, target_indices, time_indices
    )
    out_fields_app = utils.denormalize_output(
        out_fields, config["problem"], config["normalization"]
    ).numpy()
    out_fields_ref = utils.denormalize_output(
        dataset_tests["out_fields"], config["problem"], config["normalization"]
    ).numpy()

    sensor_ref = out_fields_ref[:, :, sensor_indices, :]
    sensor_app = out_fields_app[:, :, sensor_indices, :]
    nrmse = metrics.nrmse(out_fields_app, out_fields_ref)
    pearson_dissimilarity = metrics.pearson_dissimilarity(out_fields_app, out_fields_ref)
    sensor_nrmse = metrics.nrmse(sensor_app, sensor_ref)
    horizon_values = metrics.horizon_nrmse(out_fields_app, out_fields_ref)
    valid_components = loss_components_on_chunks(
        model,
        valid_chunks,
        config,
        args,
        sensor_indices,
        target_indices,
        time_indices,
    )

    if not args.skip_figures:
        save_loss_plot(history, args.adam_epochs, args.output_dir / "loss.png")
        fig = utils.plot_output_1D(
            dataset_tests, out_fields_ref, out_fields_app, 5, 4, title_ROM="JEPA-LDNet"
        )
        fig.savefig(args.output_dir / "comparison.png", dpi=200)
        plt.close(fig)

    elapsed_seconds = time.time() - start_time
    end_time_text = time.strftime("%Y-%m-%d %H:%M:%S %z")
    command = command_text()
    run_config = {
        "run_status": "completed",
        "case": args.case,
        "command": command,
        "command_hash": command_hash(command),
        "start_time": start_time_text,
        "end_time": end_time_text,
        "elapsed_seconds": elapsed_seconds,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "git_commit": current_git_commit(),
        "git_status_short": current_git_status(),
        "cuda_visible_devices": cuda_visible_devices,
        "gpu_devices": gpu_names,
        "gpu_note": (
            "TensorFlow reports devices after CUDA_VISIBLE_DEVICES remapping; "
            "cuda_visible_devices records the requested physical GPU selection."
        ),
        "sensor_indices": sensor_indices.tolist(),
        "target_indices": target_indices.tolist(),
        "target_time_indices": time_indices.tolist(),
        "target_selection": {
            "mode": args.target_mode,
            "mask_ratio": args.mask_ratio,
            "target_points": args.target_points,
            "time_strategy": args.target_time_strategy,
        },
        "model": {
            "latent_dim": config["num_latent_states"],
            "embedding_dim": args.embedding_dim,
            "condition_dim": condition_dim(config),
            "modules": [
                "ObservationEncoder_E_phi",
                "LatentTransition_T_theta",
                "ContinuousDecoder_D_omega",
                "TargetEncoder_E_bar",
                "Predictor_P_psi",
            ],
        },
        "loss": {
            "lambda_rec": args.lambda_rec,
            "lambda_jepa": args.lambda_jepa,
            "lambda_dyn": 0.0,
            "lambda_smooth": args.lambda_smooth,
            "warmup_epochs": args.jepa_warmup_epochs,
            "jepa_ramp_epochs": args.jepa_ramp_epochs,
            "ema_decay": args.ema_decay,
        },
    }
    metrics_payload = {
        "run_status": "completed",
        "case": args.case,
        "algorithm": "JEPA-LDNet",
        "nrmse": nrmse,
        "pearson_dissimilarity": pearson_dissimilarity,
        "sensor_nrmse": sensor_nrmse,
        "horizon_nrmse": horizon_values,
        "jepa_feature_mse_valid": float(valid_components["jepa"].numpy()),
        "latent_smoothness_valid": float(valid_components["smoothness"].numpy()),
        "reconstruction_mse_valid": float(valid_components["reconstruction"].numpy()),
        "loss_train_last": history.loss_train_history[-1],
        "loss_valid_last": history.loss_valid_history[-1],
        "elapsed_seconds": elapsed_seconds,
        "gpu_memory_info": gpu_memory_info(),
        "parameter_count": metrics.parameter_count(model),
        "config": run_config,
    }

    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    print("Normalized RMSE:       %1.3e" % nrmse)
    print("Pearson dissimilarity: %1.3e" % pearson_dissimilarity)
    print("Sensor NRMSE:          %1.3e" % sensor_nrmse)
    print("Elapsed seconds:       %1.1f" % elapsed_seconds)
    print("Wrote:", args.output_dir)
    return metrics_payload


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
