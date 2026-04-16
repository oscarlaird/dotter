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
    ns = torch.arange(
        0,
        params.F + 1,
        dtype=phases.dtype,
        device=phases.device,
    )
    base_log_c_n = -math.log(params.P) - 2 * math.pi**2 * ns**2 * params.sigma**2 / params.P**2
    base_mag2 = torch.exp(2 * base_log_c_n)
    weights = params.weights.to(dtype=phases.dtype, device=phases.device)
    angles = (2 * math.pi / params.P) * phases[:, None] * ns[None, :]
    real_part = weights @ torch.cos(angles)
    imag_part = weights @ torch.sin(angles)
    interference_mag2 = real_part.pow(2) + imag_part.pow(2)
    return base_mag2[0] * interference_mag2[0] + 2 * (
        base_mag2[1:] * interference_mag2[1:]
    ).sum()


def constant_phases(count: int, period: float, *, dtype: torch.dtype | None = None, device=None):
    if dtype is None:
        dtype = torch.float32
    step = period / count
    return (torch.arange(count, dtype=dtype, device=device) + 0.5) * step


def J(phases: torch.Tensor, params: TPlaceParams):
    return loss_for_phases(phases, params)


def optimize(params: TPlaceParams, initial_phases: torch.Tensor, *, max_iter: int = 100):
    phases = torch.nn.Parameter(initial_phases.clone())
    optimizer = torch.optim.LBFGS(
        [phases],
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    closure_calls = 0

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()
        loss = J(phases, params)
        loss.backward()
        return loss

    start_time = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - start_time
    state = optimizer.state[phases]
    print(
        "LBFGS converged in %d iterations (%d closure evaluations) over %.3fs"
        % (state.get("n_iter", 0), closure_calls, elapsed)
    )

    with torch.no_grad():
        phases_out = phases.clone()
    return phases_out

if __name__ == "__main__":
    print(1)
    params = TPlaceParams(
        weights=torch.tensor([0.5,0.25,0.25]),
        sigma=0.1,
        P=1.0,
        F=10,
    )
    initial_phases = constant_phases(3, params.P)
    print(f"Initial phases: {initial_phases}")
    print(f"Initial loss: {J(initial_phases, params)}")
    phases = optimize(params, initial_phases)
    print(f"Optimized phases: {phases}")
    print(f"Optimized loss: {J(phases, params)}")