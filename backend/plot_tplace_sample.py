import importlib.util
import math
import pathlib

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parent
TPLACE_PATH = ROOT / "tplace.py"
OUTPUT_PATH = ROOT / "tplace_sample_plot.png"

spec = importlib.util.spec_from_file_location("tplace", TPLACE_PATH)
tplace = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tplace)


SEED = 0
P = 1.0
K = 300
SIGMA = 0.020
F = 100
PLOT_POINTS = 2000
PLOT_OPTIMIZE_ITERS = 30


def sample_case():
    generator = torch.Generator().manual_seed(SEED)
    weight_logits = torch.randn(K, generator=generator)
    weights = torch.softmax(weight_logits, dim=0)
    initial_constant_phases = tplace.constant_phases(K, P, dtype=weights.dtype, device=weights.device)
    params = tplace.TPlaceParams(weights=weights, sigma=SIGMA, P=P, F=F)
    return params, initial_constant_phases


def weighted_bells(phases: torch.Tensor, params: tplace.TPlaceParams, x: torch.Tensor):
    dists = x[None, :] - phases[:, None]
    inv_sigma_sq = 1 / (2 * params.sigma**2)
    log_norm = -0.5 * math.log(2 * math.pi * params.sigma**2)
    densities = (
        torch.exp(-dists**2 * inv_sigma_sq + log_norm)
        + torch.exp(-(dists - params.P) ** 2 * inv_sigma_sq + log_norm)
        + torch.exp(-(dists + params.P) ** 2 * inv_sigma_sq + log_norm)
    )
    weighted = params.weights[:, None] * densities
    total = weighted.sum(dim=0)
    return weighted, total


def differential_entropy_from_pdf(pdf: torch.Tensor, x: torch.Tensor):
    return float(-torch.trapezoid(pdf * pdf.log(), x))


def optimize_for_plot(params: tplace.TPlaceParams, initial_phases: torch.Tensor, *, max_iter: int):
    return tplace.optimize(params, initial_phases, max_iter=max_iter)


def main():
    torch.set_num_threads(1)
    params, initial_constant_phases = sample_case()
    optimized_phases_10 = optimize_for_plot(
        params,
        initial_constant_phases,
        max_iter=10,
    )
    optimized_phases_30 = optimize_for_plot(
        params,
        initial_constant_phases,
        max_iter=30,
    )
    optimized_phases_100 = optimize_for_plot(
        params,
        initial_constant_phases,
        max_iter=100,
    )
    noise_entropy = 0.5 * math.log(2 * math.pi * math.e * params.sigma**2)

    x = torch.linspace(0, params.P, PLOT_POINTS, dtype=params.weights.dtype, device=params.weights.device)
    scenarios = [
        ("Initial constant", initial_constant_phases),
        ("After optimize (max_iter=10)", optimized_phases_10),
        ("After optimize (max_iter=30)", optimized_phases_30),
        ("After optimize (max_iter=100)", optimized_phases_100),
    ]

    order = torch.argsort(params.weights, descending=True)
    colors = plt.cm.nipy_spectral(torch.linspace(0.02, 0.98, K).numpy())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True, constrained_layout=True)

    for ax, (title, phases) in zip(axes.flatten(), scenarios):
        weighted, total = weighted_bells(phases, params, x)
        objective = float(tplace.J(phases, params))
        differential_entropy = differential_entropy_from_pdf(total, x)
        usable_entropy = differential_entropy - noise_entropy
        ax.stackplot(x.numpy(), weighted[order].numpy(), colors=colors, alpha=0.95, linewidth=0)
        ax.plot(x.numpy(), total.numpy(), color="black", linewidth=2.2, label="total_density")
        ax.set_title(
            f"{title}\n"
            f"frequency-domain loss={objective:.4f}\n"
            f"differential entropy={differential_entropy:.4f}\n"
            f"usable entropy={usable_entropy:.4f}\n"
            f"min phase={float(phases.min()):.5f}, max phase={float(phases.max()):.5f}"
        )
        ax.set_xlim(0.0, params.P)
        ax.set_ylabel("weighted density")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    fig.suptitle(
        f"tplace sample | seed={SEED}, K={K}, P={P}, sigma={SIGMA}, F={F}\n"
        "Real-space wrapped Gaussian visualization\n"
        "Stacked weighted components ordered by descending weight",
        fontsize=14,
    )
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
