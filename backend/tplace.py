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

def total_distrib_for_phases(phases: torch.Tensor, params: TPlaceParams):
    # here phase \in [0, P)
    ns = torch.arange(
        -params.F,
        params.F + 1,
        dtype=phases.dtype,
        device=phases.device,
    )
    logP = math.log(params.P)
    base_log_c_n = -logP - 2 * math.pi**2 * ns**2 * params.sigma**2 / params.P**2
    phase_angles = 1j * (2 * math.pi) * (phases[:, None] / params.P) * ns[None, :]
    all_log_c_n = phase_angles + base_log_c_n[None, :]
    all_c_n = torch.exp(all_log_c_n)
    return torch.einsum(
        'kf,k->f',
        all_c_n,
        params.weights.to(dtype=all_c_n.dtype, device=all_c_n.device),
    )

def loss_for_total_distrib(total_distrib: torch.Tensor):
    # justified from renyi entropy approximation
    return total_distrib.abs().pow(2).sum()

def widths_to_phases(widths: torch.Tensor):
    half_widths = 0.5 * widths
    return widths.cumsum(dim=0) - half_widths

def J(log_widths: torch.Tensor, params: TPlaceParams):
    widths = torch.softmax(log_widths, dim=0) * params.P
    phases = widths_to_phases(widths)
    total_distrib = total_distrib_for_phases(phases, params)
    return loss_for_total_distrib(total_distrib)

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