#!/usr/bin/env python3
"""Reproducible runner for LDNets TestCase 2, the 2D Navier-Stokes case."""

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
        description="Run LDNets TestCase 2 with configurable reproduction parameters."
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "case2")
    parser.add_argument("--adam-epochs", type=int, default=200)
    parser.add_argument("--bfgs-epochs", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=80)
    parser.add_argument("--valid-samples", type=int, default=20)
    parser.add_argument("--test-samples", type=int, default=10)
    parser.add_argument("--train-points", type=int, default=200)
    parser.add_argument("--valid-points", type=int, default=200)
    parser.add_argument(
        "--test-points",
        type=int,
        default=None,
        help="Optional spatial point subsampling for faster test/smoke runs.",
    )
    parser.add_argument(
        "--eval-point-batch",
        type=int,
        default=1000,
        help="Point chunk size used only for test prediction/metrics.",
    )
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--dt-base", type=float, default=5.4)
    parser.add_argument("--num-latent-states", type=int, default=1)
    parser.add_argument("--dynamics-width", type=int, default=7)
    parser.add_argument("--reconstruction-width", type=int, default=24)
    parser.add_argument("--weight-direction", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--figure-times", type=int, default=8)
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def get_problem_and_normalization(dt_base):
    problem = {
        "space": {"dimension": 2},
        "input_parameters": [],
        "input_signals": [{"name": "u"}],
        "output_fields": [{"name": "ux"}, {"name": "uy"}],
    }

    normalization = {
        "space": {"min": [0, 0], "max": [1, 1]},
        "time": {"time_constant": dt_base},
        "input_signals": {"u": {"min": -20, "max": 20}},
        "output_fields": {
            "ux": {"min": -20, "max": 20},
            "uy": {"min": -20, "max": 20},
        },
    }
    return problem, normalization


def build_networks(args, problem):
    dyn_input_shape = (
        args.num_latent_states
        + len(problem["input_parameters"])
        + len(problem["input_signals"]),
    )
    nndyn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                args.dynamics_width, activation=tf.nn.tanh, input_shape=dyn_input_shape
            ),
            tf.keras.layers.Dense(args.dynamics_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(args.num_latent_states),
        ],
        name="NNdyn",
    )

    rec_input_shape = (
        None,
        None,
        args.num_latent_states
        + len(problem["input_signals"])
        + problem["space"]["dimension"],
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
    return nndyn, nnrec


def make_model(args, problem, normalization, nndyn, nnrec):
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
        inp_signals_expanded = tf.broadcast_to(
            tf.expand_dims(dataset["inp_signals"], axis=2),
            [
                dataset["num_samples"],
                dataset["num_times"],
                dataset["num_points"],
                len(problem["input_signals"]),
            ],
        )
        output = nnrec(
            tf.concat([states_expanded, inp_signals_expanded, dataset["points_full"]], axis=3)
        )
        alpha = 0.05
        return (output**3 + alpha * output) / (1 + alpha)

    def ldnet(dataset):
        return reconstruct_output(dataset, evolve_dynamics(dataset))

    return ldnet


def get_direction(velocity, epsilon):
    return tf.math.divide(velocity, epsilon + tf.expand_dims(tf.norm(velocity, axis=3), axis=-1))


def save_loss_plot(opt, adam_epochs, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot = ax.loglog if any(i > 0 for i in opt.iterations_history) else ax.plot
    plot(opt.iterations_history, opt.loss_train_history, "o-", label="training loss")
    plot(opt.iterations_history, opt.loss_valid_history, "o-", label="validation loss")
    if adam_epochs > 0:
        ax.axvline(adam_epochs)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_comparison_plot(args, dataset_tests, out_fields_fom, out_fields_rom, output_path):
    num_times = args.figure_times
    i_sample = 0

    n_pts = int(np.sqrt(dataset_tests["points_full"].shape[2]))
    if n_pts * n_pts != dataset_tests["points_full"].shape[2]:
        print("Skipping comparison figure: test points are not a square grid.")
        return False

    points = np.asarray(dataset_tests["points_full"][i_sample, 0, :, :])
    x_grid = np.reshape(points[:, 0], (n_pts, n_pts))
    y_grid = np.reshape(points[:, 1], (n_pts, n_pts))

    v_min = np.min(out_fields_fom[i_sample, :, :, :], axis=(0, 1))
    v_max = np.max(out_fields_fom[i_sample, :, :, :], axis=(0, 1))

    times = np.linspace(0, len(dataset_tests["times"]) - 1, num=num_times, dtype=int)
    fig, axs = plt.subplots(4, num_times, figsize=(2 * num_times, 8))
    for idx_t, i_t in enumerate(times):
        axs[0, idx_t].set_title("t = %.2f" % (dataset_tests["times"][i_t] * args.dt_base))
        for i in range(2):
            levels = matplotlib.ticker.MaxNLocator(nbins=40).tick_values(v_min[i], v_max[i])
            z_fom = np.reshape(out_fields_fom[i_sample, i_t, :, i], (n_pts, n_pts))
            z_rom = np.reshape(out_fields_rom[i_sample, i_t, :, i], (n_pts, n_pts))
            axs[2 * i + 0, idx_t].contourf(
                x_grid, y_grid, z_fom, cmap="magma", levels=levels, extend="both"
            )
            axs[2 * i + 1, idx_t].contourf(
                x_grid, y_grid, z_rom, cmap="magma", levels=levels, extend="both"
            )

    axs[0, 0].set_ylabel("$v_x$ (FOM)")
    axs[1, 0].set_ylabel("$v_x$ (ROM)")
    axs[2, 0].set_ylabel("$v_y$ (FOM)")
    axs[3, 0].set_ylabel("$v_y$ (ROM)")

    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def point_slice_dataset(dataset, start, end):
    return {
        "points": dataset["points"],
        "times": dataset["times"],
        "points_full": dataset["points_full"][:, :, start:end, :],
        "inp_parameters": dataset["inp_parameters"],
        "inp_signals": dataset["inp_signals"],
        "out_fields": dataset["out_fields"][:, :, start:end, :],
        "num_points": end - start,
        "num_times": dataset["num_times"],
        "num_samples": dataset["num_samples"],
    }


def predict_in_point_chunks(ldnet, dataset, point_batch):
    if point_batch is None or point_batch <= 0 or point_batch >= dataset["num_points"]:
        return ldnet(dataset)
    outputs = []
    for start in range(0, dataset["num_points"], point_batch):
        chunk = point_slice_dataset(dataset, start, min(start + point_batch, dataset["num_points"]))
        outputs.append(ldnet(chunk))
    return tf.concat(outputs, axis=2)


def run(args):
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_names = configure_gpus()

    problem, normalization = get_problem_and_normalization(args.dt_base)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = REPO_ROOT / "data" / "NS"
    dataset_train = utils.NS_create_dataset(
        str(data_dir / "T20_80samples.npy"), np.arange(0, args.train_samples)
    )
    dataset_valid = utils.NS_create_dataset(
        str(data_dir / "T20_20samples.npy"), np.arange(0, args.valid_samples)
    )
    dataset_tests = utils.NS_create_dataset(
        str(data_dir / "T40_10samples.npy"), np.arange(0, args.test_samples)
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

    nndyn, nnrec = build_networks(args, problem)
    nndyn.summary()
    nnrec.summary()
    ldnet = make_model(args, problem, normalization, nndyn, nnrec)

    def loss(dataset, target_velocity, target_direction):
        velocity = ldnet(dataset)
        mse_velocity = tf.reduce_mean(tf.square(velocity - target_velocity))
        direction = get_direction(velocity, args.epsilon)
        mse_direction = tf.reduce_mean(tf.square(direction - target_direction))
        return mse_velocity + args.weight_direction * mse_direction

    target_direction_train = get_direction(dataset_train["out_fields"], args.epsilon)
    target_direction_valid = get_direction(dataset_valid["out_fields"], args.epsilon)
    loss_train = lambda: loss(dataset_train, dataset_train["out_fields"], target_direction_train)
    loss_valid = lambda: loss(dataset_valid, dataset_valid["out_fields"], target_direction_valid)

    trainable_variables = nndyn.variables + nnrec.variables
    opt = optimization.OptimizationProblem(trainable_variables, loss_train, loss_valid)

    print("training (Adam)...")
    opt.optimize_keras(args.adam_epochs, tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
    print("training (BFGS)...")
    opt.optimize_BFGS(args.bfgs_epochs)

    save_loss_plot(opt, args.adam_epochs, args.output_dir / "TestCase_2_loss.png")

    out_fields = predict_in_point_chunks(ldnet, dataset_tests, args.eval_point_batch)
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
        comparison_saved = save_comparison_plot(
            args,
            dataset_tests,
            out_fields_fom,
            out_fields_rom,
            args.output_dir / "TestCase_2_comparison.png",
        )

    elapsed_seconds = time.time() - start_time
    metrics = {
        "case": "2",
        "nrmse": float(nrmse),
        "pearson_dissimilarity": float(pearson_dissimilarity),
        "final_training_loss": float(opt.loss_train_history[-1].numpy()),
        "final_validation_loss": float(opt.loss_valid_history[-1].numpy()),
        "elapsed_seconds": elapsed_seconds,
        "gpu_devices": gpu_names,
        "comparison_figure_saved": comparison_saved,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
    }
    with (args.output_dir / "TestCase_2_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Elapsed seconds:       %1.1f" % elapsed_seconds)
    print("Wrote:", args.output_dir)
    return metrics


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
