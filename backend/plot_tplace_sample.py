import importlib.util
import json
import math
import pathlib
import subprocess

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parent
TPLACE_PATH = ROOT / "tplace_scipy_fast.py"
RUST_BINARY = ROOT.parent / "bayesian/target/release/plot_data"
OUTPUT_PATH = ROOT / "tplace_sample_plot.png"

spec = importlib.util.spec_from_file_location("tplace", TPLACE_PATH)
tplace = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tplace)


SEED = 0
P = 1.0
K = 300
SIGMA = 0.020
F_VALUES = [16, 32, 100]
ITER_VALUES = [10, 25, 100]
PLOT_POINTS = 2000
SHOW_FIGURE_TITLE = False


def sample_weights():
    generator = torch.Generator().manual_seed(SEED)
    weight_logits = torch.randn(K, generator=generator)
    return torch.softmax(weight_logits, dim=0)


def run_rust_optimizer(weights: torch.Tensor, f_value: int) -> dict:
    payload = json.dumps({
        "weights": weights.tolist(),
        "sigma": SIGMA,
        "period": P,
        "f": f_value,
        "iter_counts": ITER_VALUES,
    })
    result = subprocess.run(
        [str(RUST_BINARY)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def weighted_bells(phases: np.ndarray, weights: torch.Tensor, x: torch.Tensor):
    phases_t = torch.as_tensor(phases, dtype=x.dtype, device=x.device)
    dists = x[None, :] - phases_t[:, None]
    inv_sigma_sq = 1 / (2 * SIGMA**2)
    log_norm = -0.5 * math.log(2 * math.pi * SIGMA**2)
    densities = (
        torch.exp(-dists**2 * inv_sigma_sq + log_norm)
        + torch.exp(-(dists - P) ** 2 * inv_sigma_sq + log_norm)
        + torch.exp(-(dists + P) ** 2 * inv_sigma_sq + log_norm)
    )
    weighted = weights[:, None] * densities
    total = weighted.sum(dim=0)
    return weighted, total


def differential_entropy_from_pdf(pdf: torch.Tensor, x: torch.Tensor):
    return float(-torch.trapezoid(pdf * pdf.log(), x))


def style_axis(ax):
    ax.set_xlim(0.0, P)
    ax.tick_params(labelsize=12)


def add_metrics_box(ax, differential_entropy: float, usable_entropy: float):
    ax.text(
        0.96,
        0.93,
        f"differential entropy = {differential_entropy:.4f}\n"
        f"usable entropy = {usable_entropy:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        bbox={
            "boxstyle": "square,pad=0.35",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.0,
            "alpha": 0.96,
        },
    )


def main():
    weights = sample_weights()
    noise_entropy = 0.5 * math.log(2 * math.pi * math.e * SIGMA**2)

    rust_data_by_f = {f_value: run_rust_optimizer(weights, f_value) for f_value in F_VALUES}
    initial_phases = np.array(rust_data_by_f[F_VALUES[0]]["initial_phases"])
    optimized_by_f_and_iter = {
        f_value: {
            e["max_iter"]: (np.array(e["phases"]), e["loss"])
            for e in rust_data_by_f[f_value]["optimized"]
        }
        for f_value in F_VALUES
    }

    x = torch.linspace(0, P, PLOT_POINTS, dtype=weights.dtype, device=weights.device)
    order = torch.argsort(weights, descending=True)
    colors = plt.cm.nipy_spectral(torch.linspace(0.02, 0.98, K).numpy())

    fig = plt.figure(figsize=(18, 18), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], hspace=0.18)
    initial_ax = fig.add_subplot(gs[0, 1])
    grid_axes = np.empty((3, 3), dtype=object)
    for row in range(3):
        for col in range(3):
            sharex = grid_axes[0, col] if row > 0 else None
            sharey = grid_axes[row, 0] if col > 0 else None
            grid_axes[row, col] = fig.add_subplot(gs[row + 1, col], sharex=sharex, sharey=sharey)

    weighted, total = weighted_bells(initial_phases, weights, x)
    differential_entropy = differential_entropy_from_pdf(total, x)
    usable_entropy = differential_entropy - noise_entropy
    initial_ax.stackplot(
        x.numpy(),
        weighted[order].numpy(),
        colors=colors,
        alpha=0.95,
        linewidth=0.4,
        edgecolor=(0.0, 0.0, 0.0, 0.10),
    )
    initial_ax.plot(x.numpy(), total.numpy(), color="black", linewidth=2.6)
    initial_ax.set_title("Equally Spaced Phases", fontsize=16, fontweight="bold", pad=10)
    style_axis(initial_ax)
    add_metrics_box(initial_ax, differential_entropy, usable_entropy)
    initial_ax.set_ylabel("density", fontsize=16)
    initial_ax.set_xlabel("phase", fontsize=16)

    for row, f_value in enumerate(F_VALUES):
        for col, iter_value in enumerate(ITER_VALUES):
            ax = grid_axes[row, col]
            phases, _rust_loss = optimized_by_f_and_iter[f_value][iter_value]
            weighted, total = weighted_bells(phases, weights, x)
            differential_entropy = differential_entropy_from_pdf(total, x)
            usable_entropy = differential_entropy - noise_entropy
            ax.stackplot(
                x.numpy(),
                weighted[order].numpy(),
                colors=colors,
                alpha=0.95,
                linewidth=0.4,
                edgecolor=(0.0, 0.0, 0.0, 0.10),
            )
            ax.plot(x.numpy(), total.numpy(), color="black", linewidth=2.4)
            style_axis(ax)
            add_metrics_box(ax, differential_entropy, usable_entropy)
            if col == 0:
                ax.set_ylabel("density", fontsize=16)
            if row == len(F_VALUES) - 1:
                ax.set_xlabel("phase", fontsize=16)
            ax.label_outer()

    for col, iter_value in enumerate(ITER_VALUES):
        grid_axes[0, col].annotate(
            f"I = {iter_value}",
            xy=(0.5, 1.08),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    for row, f_value in enumerate(F_VALUES):
        grid_axes[row, 0].annotate(
            f"F = {f_value}",
            xy=(-0.18, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=16,
            fontweight="bold",
        )

    if SHOW_FIGURE_TITLE:
        fig.suptitle(
            f"tplace sample | seed={SEED}, K={K}, P={P}, sigma={SIGMA}\n"
            "Initial constant panel above optimized layouts across Fourier modes and iteration budgets",
            fontsize=16,
        )
    fig.savefig(OUTPUT_PATH, dpi=300)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
