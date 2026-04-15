#%%
import math
import logging
import time
from typing import NamedTuple

import torch

class TPlaceParams(NamedTuple):
    weights: torch.Tensor
    sigma: float
    P: float
    F: int


def loss_for_phases(phases: torch.Tensor, params: TPlaceParams):
    # For each nonnegative Fourier mode n, the wrapped-Gaussian coefficient is
    # a fixed Gaussian envelope term times a phase-interference term:
    #   c_n = base_c_n * sum_k w_k * exp(i * 2*pi*n*phase_k / P)
    # We only need |c_n|^2, so compute the interference power directly from
    # real cos/sin components.
    ns = torch.arange(
        0,
        params.F + 1,
        dtype=phases.dtype,
        device=phases.device,
    )
    base_log_c_n = -math.log(params.P) - 2 * math.pi**2 * ns**2 * params.sigma**2 / params.P**2
    # Real phases imply c_-n = conj(c_n), so the omitted negative frequencies
    # contribute the same squared magnitude as the positive ones.
    base_mag2 = torch.exp(2 * base_log_c_n)
    angles = (2 * math.pi / params.P) * phases[:, None] * ns[None, :]
    weights = params.weights.to(dtype=phases.dtype, device=phases.device)
    real_part = weights @ torch.cos(angles)
    imag_part = weights @ torch.sin(angles)
    interference_mag2 = real_part.pow(2) + imag_part.pow(2)
    return base_mag2[0] * interference_mag2[0] + 2 * (
        base_mag2[1:] * interference_mag2[1:]
    ).sum()


def widths_to_phases(widths: torch.Tensor):
    half_widths = 0.5 * widths
    return widths.cumsum(dim=0) - half_widths


def J(log_widths: torch.Tensor, params: TPlaceParams):
    widths = torch.softmax(log_widths, dim=0) * params.P
    phases = widths_to_phases(widths)
    return loss_for_phases(phases, params)


def optimize(params: TPlaceParams, initial_widths: torch.Tensor):
    initial_log_widths = torch.log(initial_widths)
    log_widths = torch.nn.Parameter(initial_log_widths.clone())
    optimizer = torch.optim.LBFGS(
        [log_widths],
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    closure_calls = 0

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()
        loss = J(log_widths, params)
        loss.backward()
        return loss

    start_time = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - start_time
    state = optimizer.state[log_widths]
    print(
        "LBFGS converged in %d iterations (%d closure evaluations) over %.3fs"
        % (state.get("n_iter", 0), closure_calls, elapsed)
    )

    with torch.no_grad():
        widths = torch.softmax(log_widths, dim=0) * params.P
    return widths

if __name__ == "__main__":
    print(1)
    params = TPlaceParams(
        weights=torch.tensor([0.5,0.25,0.25]),
        sigma=0.1,
        P=1.0,
        F=10,
    )
    initial_widths = torch.tensor([0.1,0.3,0.6])
    print(f"Initial widths: {initial_widths}")
    print(f"Initial loss: {J(torch.log(initial_widths), params)}")
    widths = optimize(params, initial_widths)
    print(f"Optimized widths: {widths}")
    print(f"Optimized loss: {J(torch.log(widths), params)}")