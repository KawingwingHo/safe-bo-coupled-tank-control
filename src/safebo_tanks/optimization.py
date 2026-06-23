"""Search strategies for controller auto-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, qmc
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .plant import CoupledTankPlant
from .space import ControllerSpace


@dataclass(frozen=True)
class SearchConfig:
    budget: int = 30
    num_candidates: int = 2048
    beta_objective: float = 1.8
    beta_safety: float = 3.0
    safety_buffer: float = 0.15
    expansion_weight: float = 0.25
    max_safe_step: float = 0.18
    expected_improvement_xi: float = 0.005


@dataclass
class SearchTrace:
    method: str
    seed: int
    x: list[NDArray[np.float64]] = field(default_factory=list)
    gains: list[NDArray[np.float64]] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    cost: list[float] = field(default_factory=list)
    safety: list[float] = field(default_factory=list)
    safe: list[bool] = field(default_factory=list)
    violation_steps: list[int] = field(default_factory=list)
    predicted_safety_lcb: list[float] = field(default_factory=list)

    def append(
        self,
        x: NDArray[np.float64],
        gains: NDArray[np.float64],
        objective: float,
        cost: float,
        safety: float,
        safe: bool,
        violation_steps: int,
        predicted_safety_lcb: float = np.nan,
    ) -> None:
        self.x.append(np.asarray(x, dtype=float))
        self.gains.append(np.asarray(gains, dtype=float))
        self.objective.append(float(objective))
        self.cost.append(float(cost))
        self.safety.append(float(safety))
        self.safe.append(bool(safe))
        self.violation_steps.append(int(violation_steps))
        self.predicted_safety_lcb.append(float(predicted_safety_lcb))

    def best_safe_index(self) -> int:
        indices = np.flatnonzero(np.asarray(self.safe, dtype=bool))
        if indices.size == 0:
            return 0
        local = int(np.argmin(np.asarray(self.cost)[indices]))
        return int(indices[local])

    def best_safe_gains(self) -> NDArray[np.float64]:
        return self.gains[self.best_safe_index()]


def _kernel(dimension: int) -> object:
    return ConstantKernel(1.0, (0.05, 20.0)) * Matern(
        length_scale=np.full(dimension, 0.28),
        length_scale_bounds=(0.06, 2.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.2))


def _fit_gp(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    normalize_y: bool,
    seed: int,
) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(
        kernel=_kernel(x.shape[1]),
        normalize_y=normalize_y,
        random_state=seed,
        n_restarts_optimizer=0,
        alpha=1e-6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x, y)
    return gp


def _expected_improvement(
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    best: float,
    xi: float,
) -> NDArray[np.float64]:
    improvement = mean - best - xi
    z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 1e-12)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.where(std > 1e-12, ei, 0.0)


def _candidate_pool(dimension: int, size: int, seed: int) -> NDArray[np.float64]:
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    return sampler.random(size)


def run_search(
    method: str,
    plant: CoupledTankPlant,
    space: ControllerSpace,
    seed: int,
    config: SearchConfig | None = None,
) -> SearchTrace:
    """Run one online-tuning campaign.

    ``safe_bo`` is intentionally named generically. It implements a
    confidence-bound safe set plus uncertainty-weighted boundary exploration;
    it is not presented as an exact reproduction of SafeOpt or SafeCtrlBO.
    """

    cfg = config or SearchConfig()
    valid_methods = {"manual", "random", "bo", "safe_bo"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}")
    if cfg.budget < 5:
        raise ValueError("budget must be at least the five commissioning trials")

    rng = np.random.default_rng(seed)
    candidates = _candidate_pool(space.dimension, cfg.num_candidates, seed + 911)
    initial = space.certified_initial_design()
    candidates = np.vstack([initial, candidates])
    available = np.ones(len(candidates), dtype=bool)
    trace = SearchTrace(method=method, seed=seed)

    def evaluate(x: NDArray[np.float64], trial: int, predicted_lcb: float = np.nan) -> None:
        gains = space.to_gains(x)
        # A tuning campaign uses one repeatable commissioning profile. Changing
        # the plant mismatch and leak on every query would confound controller
        # effects with scenario effects. Robustness is assessed later on held-
        # out scenarios rather than leaking that variation into the optimizer.
        rollout_seed = seed * 100_000 + 17
        result = plant.evaluate(gains, seed=rollout_seed)
        trace.append(
            x=x,
            gains=gains,
            objective=result.objective,
            cost=result.cost,
            safety=result.safety,
            safe=result.safe,
            violation_steps=result.violation_steps,
            predicted_safety_lcb=predicted_lcb,
        )

    for trial, x in enumerate(initial):
        evaluate(x, trial)
        available[trial] = False

    for trial in range(len(initial), cfg.budget):
        x_obs = np.vstack(trace.x)
        objective = np.asarray(trace.objective)
        safety = np.asarray(trace.safety)
        available_indices = np.flatnonzero(available)
        x_available = candidates[available_indices]
        predicted_lcb = np.nan

        if method == "manual":
            x_next = space.conservative_seed()
            pool_index = None
        elif method == "random":
            pool_index = int(rng.choice(available_indices))
            x_next = candidates[pool_index]
        else:
            gp_objective = _fit_gp(
                x_obs, objective, normalize_y=True, seed=seed + trial
            )
            mean_obj, std_obj = gp_objective.predict(x_available, return_std=True)

            if method == "bo":
                acquisition = _expected_improvement(
                    mean_obj,
                    std_obj,
                    best=float(np.max(objective)),
                    xi=cfg.expected_improvement_xi,
                )
                local_index = int(np.argmax(acquisition))
            else:
                gp_safety = _fit_gp(
                    x_obs, safety, normalize_y=False, seed=seed + 10_000 + trial
                )
                mean_safe, std_safe = gp_safety.predict(x_available, return_std=True)
                lcb = mean_safe - cfg.beta_safety * std_safe
                safe_observed = np.flatnonzero(safety >= 0.0)
                nearest_safe_distance = np.min(
                    cdist(x_available, x_obs[safe_observed]), axis=1
                )
                certified = (lcb >= cfg.safety_buffer) & (
                    nearest_safe_distance <= cfg.max_safe_step
                )
                if np.any(certified):
                    # Objective UCB drives performance; safety uncertainty pulls
                    # samples toward the certified boundary so the safe set can grow.
                    acquisition = (
                        mean_obj
                        + cfg.beta_objective * std_obj
                        + cfg.expansion_weight * std_safe
                    )
                    acquisition[~certified] = -np.inf
                    local_index = int(np.argmax(acquisition))
                    pool_index = int(available_indices[local_index])
                    x_next = candidates[pool_index]
                    predicted_lcb = float(lcb[local_index])
                else:
                    # If the model cannot certify a new point, repeat the
                    # observed point with the largest measured safety margin.
                    # Querying the least-bad uncertified point would violate
                    # the method's own contract.
                    fallback = int(safe_observed[np.argmax(safety[safe_observed])])
                    x_next = x_obs[fallback]
                    pool_index = None
                    predicted_lcb = np.nan

            if method == "bo":
                pool_index = int(available_indices[local_index])
                x_next = candidates[pool_index]

        evaluate(x_next, trial, predicted_lcb)
        if pool_index is not None:
            available[pool_index] = False

    return trace
