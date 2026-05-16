#!/usr/bin/env python3
"""Reproducible runner for LDNets TestCase 3, the AP1D case."""

import argparse
import json
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


def configure_gpus():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print("TensorFlow physical GPUs:", gpus)
    return [gpu.name for gpu in gpus]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run LDNets TestCase 3 with configurable reproduction parameters."
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "case3")
    parser.add_argument("--adam-epochs", type=int, default=200)
    parser.add_argument("--bfgs-epochs", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--valid-start", type=int, default=100)
    parser.add_argument("--valid-samples", type=int, default=100)
    parser.add_argument("--test-start", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--points-subsampling-rate", type=int, default=8)
    parser.add_argument("--time-steps", type=int, default=501)
    parser.add_argument("--train-points", type=int, default=20)
    parser.add_argument("--valid-points", type=int, default=20)
    parser.add_argument(
        "--test-points",
        type=int,
        default=None,
        help="Optional spatial point subsampling for faster test/smoke runs.",
    )
    parser.add_argument(
        "--eval-batch-samples",
        type=int,
        default=25,
        help="Sample chunk size used only for test prediction/metrics.",
    )
    parser.add_argument("--dt", type=float, default=1)
    parser.add_argument("--dt-base", type=float, default=205)
    parser.add_argument("--num-latent-states", type=int, default=12)
    parser.add_argument("--dynamics-width", type=int, default=8)
    parser.add_argument("--reconstruction-width", type=int, default=17)
    parser.add_argument("--alpha-reg", type=float, default=4.7e-3)
    parser.add_argument("--plot-rows", type=int, default=5)
    parser.add_argument("--plot-cols", type=int, default=4)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def get_problem_and_normalization(dt_base):
    problem = {
        "space": {"dimension": 1},
        "input_parameters": [],
        "input_signals": [{"name": "Iapp1"}, {"name": "Iapp2"}],
        "output_fields": [{"name": "u"}],
    }

    normalization = {
        "space": {"min": [0.0], "max": [100.0]},
        "time": {"time_constant": dt_base},
        "input_signals": {
            "Iapp1": {"min": -2.5, "max": 2.5},
            "Iapp2": {"min": -2.5, "max": 2.5},
        },
        "output_fields": {"u": {"min": 0.0, "max": 1.2}},
    }
    return problem, normalization


def sample_range(start, count):
    return np.arange(start, start + count)


def build_networks(args, problem):
    dyn_input_shape = (
        args.num_latent_states
        + len(problem["input_parameters"])
        + len(problem["input_signals"]),
    )
    nndyn_base = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                args.dynamics_width, activation=tf.nn.tanh, input_shape=dyn_input_shape
            ),
            tf.keras.layers.Dense(args.num_latent_states),
        ],
        name="NNdyn_base",
    )

    rec_input_shape = (
        None,
        None,
        args.num_latent_states + problem["space"]["dimension"],
    )
    nnrec = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                args.reconstruction_width,
                activation=tf.nn.tanh,
                input_shape=rec_input_shape,
            ),
            tf.keras.layers.Dense(args.reconstruction_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(args.reconstruction_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(args.reconstruction_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(args.reconstruction_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(len(problem["output_fields"])),
        ],
        name="NNrec",
    )
    return nndyn_base, nnrec


def make_model(args, problem, normalization, nndyn_base, nnrec):
    inp_eq = tf.zeros(
        (1, args.num_latent_states + len(problem["input_signals"])), dtype=tf.float64
    )

    def nndyn(inp):
        return nndyn_base(inp) - nndyn_base(inp_eq)

    def evolve_dynamics(dataset):
        state = tf.zeros((dataset["num_samples"], args.num_latent_states), dtype=tf.float64)
        state_history = tf.TensorArray(tf.float64, size=dataset["num_times"])
        state_history = state_history.write(0, state)
        dt_ref = normalization["time"]["time_constant"]

        for i in tf.range(dataset["num_times"] - 1):
            state = state + args.dt / dt_ref * nndyn(
                tf.concat([state, dataset["inp_signals"][:, i, :]], axis=-1)
            )
            state_history = state_history.write(i + 1, state)

        return tf.transpose(state_history.stack(), perm=(1, 0, 2))

    def reconstruct_output(dataset, states):
        states_expanded = tf.broadcast_to(
            tf.expand_dims(states, axis=2),
            [
                dataset["num_samples"],
                dataset["num_times"],
                dataset["num_points"],
                args.num_latent_states,
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


def sample_slice_dataset(dataset, start, end):
    return {
        "points": dataset["points"],
        "times": dataset["times"],
        "points_full": dataset["points_full"][start:end],
        "inp_parameters": dataset["inp_parameters"],
        "inp_signals": dataset["inp_signals"][start:end],
        "out_fields": dataset["out_fields"][start:end],
        "num_points": dataset["num_points"],
        "num_times": dataset["num_times"],
        "num_samples": end - start,
    }


def predict_in_sample_chunks(ldnet, dataset, batch_samples):
    if batch_samples is None or batch_samples <= 0 or batch_samples >= dataset["num_samples"]:
        return ldnet(dataset)
    outputs = []
    for start in range(0, dataset["num_samples"], batch_samples):
        chunk = sample_slice_dataset(
            dataset, start, min(start + batch_samples, dataset["num_samples"])
        )
        outputs.append(ldnet(chunk))
    return tf.concat(outputs, axis=0)


def run(args):
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_names = configure_gpus()

    problem, normalization = get_problem_and_normalization(args.dt_base)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_set_path = str(REPO_ROOT / "data" / "AP1D")
    dataset_train = utils.AP_create_dataset(
        data_set_path,
        sample_range(args.train_start, args.train_samples),
        points_subsampling_rate=args.points_subsampling_rate,
        time_steps=args.time_steps,
    )
    dataset_valid = utils.AP_create_dataset(
        data_set_path,
        sample_range(args.valid_start, args.valid_samples),
        points_subsampling_rate=args.points_subsampling_rate,
        time_steps=args.time_steps,
    )
    dataset_tests = utils.AP_create_dataset(
        data_set_path,
        sample_range(args.test_start, args.test_samples),
        points_subsampling_rate=args.points_subsampling_rate,
        time_steps=args.time_steps,
    )

    utils.process_dataset(
        dataset_train,
        problem,
        normalization,
        dt=args.dt,
        num_points_subsample=args.train_points,
    )
    utils.process_dataset(
        dataset_valid,
        problem,
        normalization,
        dt=args.dt,
        num_points_subsample=args.valid_points,
    )
    utils.process_dataset(
        dataset_tests,
        problem,
        normalization,
        dt=args.dt,
        num_points_subsample=args.test_points,
    )

    nndyn_base, nnrec = build_networks(args, problem)
    nndyn_base.summary()
    nnrec.summary()
    ldnet = make_model(args, problem, normalization, nndyn_base, nnrec)

    def mse(dataset):
        out_fields = ldnet(dataset)
        error = out_fields - dataset["out_fields"]
        return tf.reduce_mean(tf.square(error))

    def loss():
        return mse(dataset_train) + args.alpha_reg * (
            weights_reg(nndyn_base) + weights_reg(nnrec)
        )

    def mse_valid():
        return mse(dataset_valid)

    trainable_variables = nndyn_base.variables + nnrec.variables
    opt = optimization.OptimizationProblem(trainable_variables, loss, mse_valid)

    print("training (Adam)...")
    opt.optimize_keras(args.adam_epochs, tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
    print("training (BFGS)...")
    opt.optimize_BFGS(args.bfgs_epochs)

    save_loss_plot(opt, args.adam_epochs, args.output_dir / "TestCase_3_loss.png")

    out_fields = predict_in_sample_chunks(ldnet, dataset_tests, args.eval_batch_samples)
    out_fields_fom = utils.denormalize_output(
        dataset_tests["out_fields"], problem, normalization
    ).numpy()
    out_fields_rom = utils.denormalize_output(out_fields, problem, normalization).numpy()

    nrmse = np.sqrt(np.mean(np.square(out_fields_rom - out_fields_fom))) / (
        np.max(out_fields_fom) - np.min(out_fields_fom)
    )
    r_coeff = scipy.stats.pearsonr(
        np.reshape(out_fields_rom, (-1,)), np.reshape(out_fields_fom, (-1,))
    )
    pearson_dissimilarity = 1 - r_coeff[0]

    print("Normalized RMSE:       %1.3e" % nrmse)
    print("Pearson dissimilarity: %1.3e" % pearson_dissimilarity)

    comparison_saved = False
    if not args.skip_figures:
        plot_rows = min(args.plot_rows, args.test_samples)
        plot_cols = min(args.plot_cols, max(1, args.test_samples // plot_rows))
        fig = utils.plot_output_1D(dataset_tests, out_fields_fom, out_fields_rom, plot_rows, plot_cols)
        fig.savefig(args.output_dir / "TestCase_3_comparison.png", dpi=200)
        plt.close(fig)
        comparison_saved = True

    elapsed_seconds = time.time() - start_time
    metrics = {
        "case": "3",
        "nrmse": float(nrmse),
        "pearson_dissimilarity": float(pearson_dissimilarity),
        "final_training_loss": float(opt.loss_train_history[-1].numpy()),
        "final_validation_loss": float(opt.loss_valid_history[-1].numpy()),
        "elapsed_seconds": elapsed_seconds,
        "gpu_devices": gpu_names,
        "comparison_figure_saved": comparison_saved,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
    }
    with (args.output_dir / "TestCase_3_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Elapsed seconds:       %1.1f" % elapsed_seconds)
    print("Wrote:", args.output_dir)
    return metrics


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
