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
    initial_constant_widths = torch.full((K,), P / K)
    params = tplace.TPlaceParams(weights=weights, sigma=SIGMA, P=P, F=F)
    return params, initial_constant_widths


def schroeder_power_widths(params: tplace.TPlaceParams):
    powers = params.weights.pow(2)
    return params.P * powers / powers.sum()


def weighted_bells(widths: torch.Tensor, params: tplace.TPlaceParams, x: torch.Tensor):
    phases = tplace.widths_to_phases(widths)
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


def optimize_for_plot(params: tplace.TPlaceParams, initial_widths: torch.Tensor, *, max_iter: int):
    initial_log_widths = torch.log(initial_widths)
    log_widths = torch.nn.Parameter(initial_log_widths.clone())
    optimizer = torch.optim.LBFGS(
        [log_widths],
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = tplace.J(log_widths, params)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.softmax(log_widths, dim=0) * params.P


def main():
    torch.set_num_threads(1)
    params, initial_constant_widths = sample_case()
    initial_power_widths = schroeder_power_widths(params)
    optimized_constant_widths = optimize_for_plot(
        params,
        initial_constant_widths,
        max_iter=PLOT_OPTIMIZE_ITERS,
    )
    optimized_power_widths = optimize_for_plot(
        params,
        initial_power_widths,
        max_iter=PLOT_OPTIMIZE_ITERS,
    )
    noise_entropy = 0.5 * math.log(2 * math.pi * math.e * params.sigma**2)

    x = torch.linspace(0, params.P, PLOT_POINTS, dtype=params.weights.dtype, device=params.weights.device)
    scenarios = [
        ("Initial constant", initial_constant_widths),
        ("Schroeder-style init (widths ∝ weights^2)", initial_power_widths),
        (f"Constant + optimize ({PLOT_OPTIMIZE_ITERS})", optimized_constant_widths),
        (f"Power init + optimize ({PLOT_OPTIMIZE_ITERS})", optimized_power_widths),
    ]

    order = torch.argsort(params.weights, descending=True)
    colors = plt.cm.nipy_spectral(torch.linspace(0.02, 0.98, K).numpy())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True, constrained_layout=True)

    for ax, (title, widths) in zip(axes.flatten(), scenarios):
        weighted, total = weighted_bells(widths, params, x)
        objective = float(tplace.J(torch.log(widths), params))
        differential_entropy = differential_entropy_from_pdf(total, x)
        usable_entropy = differential_entropy - noise_entropy
        ax.stackplot(x.numpy(), weighted[order].numpy(), colors=colors, alpha=0.95, linewidth=0)
        ax.plot(x.numpy(), total.numpy(), color="black", linewidth=2.2, label="total_density")
        ax.set_title(
            f"{title}\n"
            f"frequency-domain loss={objective:.4f}\n"
            f"differential entropy={differential_entropy:.4f}\n"
            f"usable entropy={usable_entropy:.4f}\n"
            f"min width={float(widths.min()):.5f}, max width={float(widths.max()):.5f}"
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
