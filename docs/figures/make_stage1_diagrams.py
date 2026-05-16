#!/usr/bin/env python3
"""Generate deterministic Stage 1 JEPA-LDNet diagram fallbacks."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent


def add_box(ax, xy, wh, text, face="#f7f9fb", edge="#2f3b52", fontsize=9.5):
    rect = Rectangle(
        xy,
        wh[0],
        wh[1],
        facecolor=face,
        edgecolor=edge,
        linewidth=1.4,
        joinstyle="round",
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#182235",
        wrap=True,
    )
    return rect


def arrow(ax, start, end, color="#43506a", rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title(
        "JEPA-LDNet: Joint-Embedding Predictive Latent Dynamics Network",
        fontsize=16,
        weight="bold",
        pad=16,
    )

    top_y = 5.2
    bot_y = 2.0
    loss_y = 0.75
    w = 1.7
    h = 0.8

    boxes = {
        "obs": add_box(ax, (0.4, top_y), (w, h), "Sparse\nObservations", "#eaf2ff"),
        "enc": add_box(ax, (2.5, top_y), (w, h), "Observation\nEncoder E_phi", "#e7f7ef"),
        "z0": add_box(ax, (4.6, top_y), (w, h), "Initial Latent\nState z0", "#f7f0df"),
        "dyn": add_box(ax, (6.7, top_y), (w, h), "Latent Transition\nT_theta", "#e7f7ef"),
        "roll": add_box(ax, (8.8, top_y), (w, h), "Latent Rollout\nz_1:T", "#f7f0df"),
        "dec": add_box(ax, (10.9, top_y), (w, h), "Continuous Decoder\nD_omega", "#e7f7ef"),
        "predfield": add_box(ax, (13.0, top_y), (w, h), "Field Prediction\nY_hat", "#eaf2ff"),
        "patch": add_box(ax, (2.5, bot_y), (w, h), "Target\nPatches", "#fff1f0"),
        "tenc": add_box(ax, (4.6, bot_y), (w, h), "EMA Target\nEncoder E_bar", "#fff1f0"),
        "ht": add_box(ax, (6.7, bot_y), (w, h), "Target Embedding\nh_target", "#f7f0df"),
        "pred": add_box(ax, (9.25, bot_y), (w, h), "Predictor\nP_psi", "#e7f7ef"),
        "hp": add_box(ax, (11.35, bot_y), (w, h), "Predicted Embedding\nh_pred", "#f7f0df"),
        "jepa": add_box(ax, (9.65, loss_y), (w, h), "JEPA\nLoss", "#fdebd3"),
        "rec": add_box(ax, (13.0, 3.55), (w, h), "Reconstruction\nLoss", "#fdebd3"),
    }

    top_order = ["obs", "enc", "z0", "dyn", "roll", "dec", "predfield"]
    for left, right in zip(top_order[:-1], top_order[1:]):
        l = boxes[left]
        r = boxes[right]
        arrow(ax, (l.get_x() + w, l.get_y() + h / 2), (r.get_x(), r.get_y() + h / 2))

    for left, right in [("patch", "tenc"), ("tenc", "ht"), ("pred", "hp")]:
        l = boxes[left]
        r = boxes[right]
        arrow(ax, (l.get_x() + w, l.get_y() + h / 2), (r.get_x(), r.get_y() + h / 2))

    arrow(ax, (9.65, top_y), (10.1, bot_y + h), color="#2c7a7b")
    arrow(ax, (7.55, bot_y), (10.0, loss_y + h), color="#9a3412", rad=-0.08)
    arrow(ax, (12.2, bot_y), (11.0, loss_y + h), color="#9a3412", rad=0.08)
    arrow(ax, (13.85, top_y), (13.85, 4.35), color="#9a3412")
    arrow(ax, (5.45, top_y), (5.45, bot_y + h), color="#6b7280", rad=-0.25)
    ax.text(5.7, 3.7, "EMA Update", fontsize=10, color="#4b5563")
    ax.text(7.4, 1.55, "stop-gradient target", fontsize=9, color="#6b7280")

    save(fig, "stage1_jepa_ldnet_architecture")


def training_flow():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("JEPA-LDNet Training Flow", fontsize=16, weight="bold", pad=16)

    left_x = 0.7
    mid_x = 5.1
    right_x = 9.5
    w = 3.2
    h = 0.65
    ys = [6.4, 5.35, 4.3, 3.25, 2.2]

    b1 = add_box(ax, (left_x, ys[0]), (w, h), "1. Load Normalized\nField Data", "#eaf2ff")
    b2 = add_box(ax, (left_x, ys[1]), (w, h), "2. Sample Context\nSensors", "#eaf2ff")
    b3 = add_box(ax, (left_x, ys[2]), (w, h), "3. Encode Latent\nState", "#e7f7ef")
    b4 = add_box(ax, (left_x, ys[3]), (w, h), "4. Roll Out Latent\nDynamics", "#e7f7ef")
    b5 = add_box(ax, (left_x, ys[4]), (w, h), "5. Decode Query\nField", "#e7f7ef")

    b6 = add_box(ax, (mid_x, ys[1]), (w, h), "6. Build JEPA\nTargets", "#fff1f0")
    b7 = add_box(ax, (mid_x, ys[2]), (w, h), "7. Predict Target\nEmbeddings", "#fff1f0")
    b8 = add_box(ax, (mid_x, ys[3]), (w, h), "8. Compute Total Loss\nL_rec + L_jepa + L_smooth", "#fdebd3")

    b9 = add_box(ax, (right_x, ys[3]), (w, h), "9. Update Online\nNetworks", "#f7f0df")
    b10 = add_box(ax, (right_x, ys[2]), (w, h), "10. EMA Update\nTarget Encoder", "#f7f0df")
    b11 = add_box(ax, (right_x, ys[1]), (w, h), "11. Evaluate Metrics\nNRMSE | Pearson | Sensor | Horizon", "#eaf2ff", fontsize=9)

    chain = [b1, b2, b3, b4, b5]
    for upper, lower in zip(chain[:-1], chain[1:]):
        arrow(
            ax,
            (upper.get_x() + w / 2, upper.get_y()),
            (lower.get_x() + w / 2, lower.get_y() + h),
        )

    arrow(ax, (b1.get_x() + w, b1.get_y() + h / 2), (b6.get_x(), b6.get_y() + h / 2))
    arrow(ax, (b4.get_x() + w, b4.get_y() + h / 2), (b7.get_x(), b7.get_y() + h / 2))
    arrow(ax, (b5.get_x() + w, b5.get_y() + h / 2), (b8.get_x(), b8.get_y() + h / 2))
    arrow(ax, (b6.get_x() + w / 2, b6.get_y()), (b8.get_x() + w / 2, b8.get_y() + h))
    arrow(ax, (b7.get_x() + w / 2, b7.get_y()), (b8.get_x() + w / 2, b8.get_y() + h))
    arrow(ax, (b8.get_x() + w, b8.get_y() + h / 2), (b9.get_x(), b9.get_y() + h / 2))
    arrow(ax, (b9.get_x() + w / 2, b9.get_y() + h), (b10.get_x() + w / 2, b10.get_y()))
    arrow(ax, (b10.get_x() + w / 2, b10.get_y() + h), (b11.get_x() + w / 2, b11.get_y()))

    ax.text(6.0, 2.7, "Loss terms: L_rec, L_jepa, L_smooth", fontsize=10, color="#4b5563")

    save(fig, "stage1_jepa_ldnet_training_flow")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    architecture()
    training_flow()
    print("Wrote Stage 1 diagrams to", OUT_DIR)


if __name__ == "__main__":
    main()
