#!/usr/bin/env python3
"""Reproducible runner for the ADR notebook experiments TestCase 1a/1b/1c."""

import argparse
import copy
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
import scipy.stats
import tensorflow as tf

import optimization
import utils


tf.keras.backend.set_floatx("float64")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


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


CASE_CONFIGS = {
    "1a": {
        "data_path": REPO_ROOT / "data" / "ADR" / "data_1a.npy",
        "problem": {
            "space": {"dimension": 1},
            "input_parameters": [
                {"name": "coeff_diffusion"},
                {"name": "coeff_transport"},
                {"name": "coeff_reaction"},
            ],
            "input_signals": [],
            "output_fields": [{"name": "z"}],
        },
        "normalization": {
            "space": {"min": [-1], "max": [+1]},
            "time": {"time_constant": 5e-1},
            "input_parameters": {
                "coeff_diffusion": {"min": 0, "max": 5e-2},
                "coeff_transport": {"min": -1e-1, "max": 1e-1},
                "coeff_reaction": {"min": 0, "max": 1e-2},
            },
            "output_fields": {"z": {"min": -1, "max": +1}},
        },
        "dt": 5e-2,
        "num_latent_states": 2,
        "dynamics_width": 9,
        "reconstruction_width": 11,
        "adam_epochs": 50,
        "bfgs_epochs": 150,
        "alpha_reg": None,
        "driver": "parameters",
        "reference": {"nrmse": 8.576e-3, "pearson_dissimilarity": 6.835e-4},
    },
    "1b": {
        "data_path": REPO_ROOT / "data" / "ADR" / "data_1b.npy",
        "problem": {
            "space": {"dimension": 1},
            "input_parameters": [],
            "input_signals": [{"name": "amplitude"}, {"name": "phase"}],
            "output_fields": [{"name": "z"}],
        },
        "normalization": {
            "space": {"min": [-1], "max": [+1]},
            "time": {"time_constant": 2.3},
            "input_signals": {
                "amplitude": {"min": 0, "max": 0.8},
                "phase": {"min": -4, "max": 4},
            },
            "output_fields": {"z": {"min": -1, "max": +1}},
        },
        "dt": 5e-2,
        "num_latent_states": 2,
        "dynamics_width": 10,
        "reconstruction_width": 7,
        "adam_epochs": 50,
        "bfgs_epochs": 150,
        "alpha_reg": 1e-5,
        "driver": "signals",
        "reference": {"nrmse": 2.440e-2, "pearson_dissimilarity": 1.052e-2},
    },
    "1c": {
        "data_path": REPO_ROOT / "data" / "ADR" / "data_1c_fmax1.0.npy",
        "problem": {
            "space": {"dimension": 1},
            "input_parameters": [],
            "input_signals": [
                {"name": "amplitude"},
                {"name": "phase"},
                {"name": "frequency"},
            ],
            "output_fields": [{"name": "z"}],
        },
        "normalization": {
            "space": {"min": [-1], "max": [+1]},
            "time": {"time_constant": 8},
            "input_signals": {
                "amplitude": {"min": 0, "max": 2},
                "phase": {"min": -4, "max": 4},
                "frequency": {"min": 0.25, "max": 1},
            },
            "output_fields": {"z": {"min": -4, "max": +4}},
        },
        "dt": 5e-2,
        "num_latent_states": 4,
        "dynamics_width": 16,
        "reconstruction_width": 8,
        "adam_epochs": 200,
        "bfgs_epochs": 1800,
        "alpha_reg": 2.7e-4,
        "driver": "signals",
        "reference": {"nrmse": 2.039e-2, "pearson_dissimilarity": 1.152e-2},
    },
}


def config_with_overrides(config, args):
    config = copy.deepcopy(config)
    overrides = {
        "num_latent_states": args.num_latent_states,
        "dynamics_width": args.dynamics_width,
        "reconstruction_width": args.reconstruction_width,
        "alpha_reg": args.alpha_reg,
    }
    applied = {}
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
            applied[key] = value
    return config, applied


def configure_gpus():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print("TensorFlow physical GPUs:", gpus)
    return [gpu.name for gpu in gpus]


def load_case_data(config):
    dataset_raw = np.load(config["data_path"], allow_pickle=True)[()]
    dataset_train = utils.ADR_create_dataset(dataset_raw, np.arange(0, 100))
    dataset_valid = utils.ADR_create_dataset(dataset_raw, np.arange(100, 200))
    dataset_tests = utils.ADR_create_dataset(dataset_raw, np.arange(200, 300))

    utils.process_dataset(
        dataset_train, config["problem"], config["normalization"], dt=config["dt"]
    )
    utils.process_dataset(
        dataset_valid, config["problem"], config["normalization"], dt=config["dt"]
    )
    utils.process_dataset(
        dataset_tests, config["problem"], config["normalization"], dt=config["dt"]
    )
    return dataset_train, dataset_valid, dataset_tests


def build_networks(config):
    problem = config["problem"]
    num_latent_states = config["num_latent_states"]

    input_shape = (
        num_latent_states
        + len(problem["input_parameters"])
        + len(problem["input_signals"]),
    )
    nndyn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                config["dynamics_width"], activation=tf.nn.tanh, input_shape=input_shape
            ),
            tf.keras.layers.Dense(config["dynamics_width"], activation=tf.nn.tanh),
            tf.keras.layers.Dense(num_latent_states),
        ],
        name="NNdyn",
    )

    input_shape = (None, None, num_latent_states + problem["space"]["dimension"])
    nnrec = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                config["reconstruction_width"],
                activation=tf.nn.tanh,
                input_shape=input_shape,
            ),
            tf.keras.layers.Dense(config["reconstruction_width"], activation=tf.nn.tanh),
            tf.keras.layers.Dense(len(problem["output_fields"])),
        ],
        name="NNrec",
    )
    return nndyn, nnrec


def make_model(config, nndyn, nnrec):
    dt = config["dt"]
    normalization = config["normalization"]
    num_latent_states = config["num_latent_states"]

    def evolve_dynamics(dataset):
        state = tf.zeros((dataset["num_samples"], num_latent_states), dtype=tf.float64)
        state_history = tf.TensorArray(tf.float64, size=dataset["num_times"])
        state_history = state_history.write(0, state)
        dt_ref = normalization["time"]["time_constant"]

        for i in tf.range(dataset["num_times"] - 1):
            if config["driver"] == "parameters":
                dynamics_input = tf.concat([state, dataset["inp_parameters"]], axis=-1)
            else:
                dynamics_input = tf.concat([state, dataset["inp_signals"][:, i, :]], axis=-1)
            state = state + dt / dt_ref * nndyn(dynamics_input)
            state_history = state_history.write(i + 1, state)

        return tf.transpose(state_history.stack(), perm=(1, 0, 2))

    def reconstruct_output(dataset, states):
        states_expanded = tf.broadcast_to(
            tf.expand_dims(states, axis=2),
            [
                dataset["num_samples"],
                dataset["num_times"],
                dataset["num_points"],
                num_latent_states,
            ],
        )
        return nnrec(tf.concat([states_expanded, dataset["points_full"]], axis=3))

    def ldnet(dataset):
        return reconstruct_output(dataset, evolve_dynamics(dataset))

    return ldnet


def weights_reg(network):
    return sum(tf.reduce_mean(tf.square(layer.kernel)) for layer in network.layers) / len(
        network.layers
    )


def slice_dataset(dataset, start, end):
    return {
        "points": dataset["points"],
        "times": dataset["times"],
        "points_full": dataset["points_full"][start:end],
        "inp_parameters": (
            None
            if dataset["inp_parameters"] is None
            else dataset["inp_parameters"][start:end]
        ),
        "inp_signals": (
            None if dataset["inp_signals"] is None else dataset["inp_signals"][start:end]
        ),
        "out_fields": dataset["out_fields"][start:end],
        "num_points": dataset["num_points"],
        "num_times": dataset["num_times"],
        "num_samples": end - start,
    }


def dataset_chunks(dataset, batch_samples):
    if batch_samples is None or batch_samples <= 0:
        return [dataset]
    return [
        slice_dataset(dataset, start, min(start + batch_samples, dataset["num_samples"]))
        for start in range(0, dataset["num_samples"], batch_samples)
    ]


def save_loss_plot(opt, adam_epochs, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot = ax.loglog if any(i > 0 for i in opt.iterations_history) else ax.plot
    plot(opt.iterations_history, opt.loss_train_history, "o-", label="training loss")
    plot(opt.iterations_history, opt.loss_valid_history, "o-", label="validation loss")
    if adam_epochs > 0:
        ax.axvline(adam_epochs)
    ax.set_xlabel("epochs")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_case(
    case_name,
    output_dir,
    args,
    adam_epochs=None,
    bfgs_epochs=None,
    learning_rate=1e-2,
    seed=0,
    batch_samples=25,
):
    start_time = time.time()
    config, config_overrides = config_with_overrides(CASE_CONFIGS[case_name], args)
    output_dir.mkdir(parents=True, exist_ok=True)
    adam_epochs = config["adam_epochs"] if adam_epochs is None else adam_epochs
    bfgs_epochs = config["bfgs_epochs"] if bfgs_epochs is None else bfgs_epochs

    gpu_names = configure_gpus()

    print(f"\n===== TestCase {case_name} =====")
    print("data:", config["data_path"])

    np.random.seed(seed)
    tf.random.set_seed(seed)

    dataset_train, dataset_valid, dataset_tests = load_case_data(config)
    train_chunks = dataset_chunks(dataset_train, batch_samples)
    valid_chunks = dataset_chunks(dataset_valid, batch_samples)
    test_chunks = dataset_chunks(dataset_tests, batch_samples)
    nndyn, nnrec = build_networks(config)
    nndyn.summary()
    nnrec.summary()
    ldnet = make_model(config, nndyn, nnrec)

    def mse(chunks):
        sum_squared_error = tf.constant(0.0, dtype=tf.float64)
        num_entries = tf.constant(0.0, dtype=tf.float64)
        for chunk in chunks:
            error = ldnet(chunk) - chunk["out_fields"]
            sum_squared_error += tf.reduce_sum(tf.square(error))
            num_entries += tf.cast(tf.size(error), tf.float64)
        return sum_squared_error / num_entries

    if config["alpha_reg"] is None:
        def loss():
            return mse(train_chunks)
    else:
        def loss():
            return mse(train_chunks) + config["alpha_reg"] * (
                weights_reg(nndyn) + weights_reg(nnrec)
            )

    def mse_valid():
        return mse(valid_chunks)

    trainable_variables = nndyn.variables + nnrec.variables
    opt = optimization.OptimizationProblem(trainable_variables, loss, mse_valid)

    if adam_epochs > 0:
        print("training (Adam)...")
        opt.optimize_keras(adam_epochs, tf.keras.optimizers.Adam(learning_rate=learning_rate))
    if bfgs_epochs > 0:
        print("training (BFGS)...")
        opt.optimize_BFGS(bfgs_epochs)

    out_fields = tf.concat([ldnet(chunk) for chunk in test_chunks], axis=0)
    out_fields_app = utils.denormalize_output(
        out_fields, config["problem"], config["normalization"]
    ).numpy()
    out_fields_ref = utils.denormalize_output(
        dataset_tests["out_fields"], config["problem"], config["normalization"]
    ).numpy()

    nrmse = np.sqrt(np.mean(np.square(out_fields_app - out_fields_ref))) / (
        np.max(out_fields_ref) - np.min(out_fields_ref)
    )
    r_coeff = scipy.stats.pearsonr(
        np.reshape(out_fields_app, (-1,)), np.reshape(out_fields_ref, (-1,))
    )
    pearson_dissimilarity = 1 - r_coeff[0]

    print("Normalized RMSE:       %1.3e" % nrmse)
    print("Pearson dissimilarity: %1.3e" % pearson_dissimilarity)
    elapsed_seconds = time.time() - start_time
    command = command_text()

    save_loss_plot(
        opt, adam_epochs, output_dir / f"TestCase_{case_name}_loss.png"
    )
    fig = utils.plot_output_1D(
        dataset_tests, out_fields_ref, out_fields_app, 5, 4, title_ROM="LDNet"
    )
    fig.savefig(output_dir / f"TestCase_{case_name}_comparison.png", dpi=200)
    plt.close(fig)

    result = {
        "run_status": "completed",
        "case": case_name,
        "algorithm": "Original-LDNet",
        "nrmse": float(nrmse),
        "pearson_dissimilarity": float(pearson_dissimilarity),
        "reference": config["reference"],
        "adam_epochs": adam_epochs,
        "bfgs_epochs": bfgs_epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "batch_samples": batch_samples,
        "data_path": str(config["data_path"]),
        "dt": config["dt"],
        "normalization": config["normalization"],
        "config_overrides": config_overrides,
        "model": {
            "latent_dim": config["num_latent_states"],
            "dynamics_width": config["dynamics_width"],
            "reconstruction_width": config["reconstruction_width"],
            "alpha_reg": config["alpha_reg"],
        },
        "loss_train_last": float(opt.loss_train_history[-1].numpy()),
        "loss_valid_last": float(opt.loss_valid_history[-1].numpy()),
        "elapsed_seconds": elapsed_seconds,
        "command": command,
        "command_hash": command_hash(command),
        "git_commit": current_git_commit(),
        "git_status_short": current_git_status(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_devices": gpu_names,
    }
    result_path = output_dir / f"TestCase_{case_name}_metrics.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["1a", "1b", "1c", "all"],
        default="all",
        help="ADR notebook case to run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runs" / "case1"),
        help="Directory for metrics and figures.",
    )
    parser.add_argument(
        "--adam-epochs",
        type=int,
        default=None,
        help="Override Adam epochs for selected cases.",
    )
    parser.add_argument(
        "--bfgs-epochs",
        type=int,
        default=None,
        help="Override BFGS epochs for selected cases.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-samples",
        type=int,
        default=25,
        help="Split full-batch loss/prediction by sample count; use 0 for no split.",
    )
    parser.add_argument("--num-latent-states", type=int, default=None)
    parser.add_argument("--dynamics-width", type=int, default=None)
    parser.add_argument("--reconstruction-width", type=int, default=None)
    parser.add_argument("--alpha-reg", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    cases = ["1a", "1b", "1c"] if args.case == "all" else [args.case]
    output_dir = Path(args.output_dir).resolve()
    new_results = [
        run_case(
            case_name,
            output_dir,
            args,
            adam_epochs=args.adam_epochs,
            bfgs_epochs=args.bfgs_epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            batch_samples=args.batch_samples,
        )
        for case_name in cases
    ]
    all_results = {result["case"]: result for result in new_results}
    for metrics_path in sorted(output_dir.glob("TestCase_1?_metrics.json")):
        result = json.loads(metrics_path.read_text())
        all_results[result["case"]] = result
    (output_dir / "summary_metrics.json").write_text(
        json.dumps([all_results[case] for case in sorted(all_results)], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
