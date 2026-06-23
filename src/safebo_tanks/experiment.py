"""Reproducible experiment runner and result visualization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .optimization import SearchConfig, SearchTrace, run_search
from .plant import CoupledTankPlant
from .space import ControllerSpace


METHODS = ("manual", "random", "bo", "safe_bo")
LABELS = {
    "manual": "Conservative PI",
    "random": "Random search",
    "bo": "Ordinary BO",
    "safe_bo": "Safe BO",
}
COLORS = {
    "manual": "#6b7280",
    "random": "#d97706",
    "bo": "#dc2626",
    "safe_bo": "#059669",
}


def _trace_rows(trace: SearchTrace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    best = np.inf
    cumulative_unsafe = 0
    for trial in range(len(trace.cost)):
        if trace.safe[trial]:
            best = min(best, trace.cost[trial])
        else:
            cumulative_unsafe += 1
        row: dict[str, object] = {
            "method": trace.method,
            "seed": trace.seed,
            "trial": trial + 1,
            "cost": trace.cost[trial],
            "safety_margin": trace.safety[trial],
            "safe": trace.safe[trial],
            "violation_steps": trace.violation_steps[trial],
            "best_safe_cost": best if np.isfinite(best) else np.nan,
            "cumulative_unsafe_trials": cumulative_unsafe,
            "predicted_safety_lcb": trace.predicted_safety_lcb[trial],
        }
        for name, value in zip(("Kp1", "Ki1", "Kp2", "Ki2"), trace.gains[trial]):
            row[name] = value
        rows.append(row)
    return rows


def _qualify_controller(
    trace: SearchTrace,
    plant: CoupledTankPlant,
    qualification_scenarios: int,
    top_k: int = 10,
    minimum_observed_margin: float = 0.10,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select a deployment controller using held-out digital-twin scenarios.

    This gate is deliberately separate from online optimization. GP confidence
    is model-dependent; a controller must also pass several parameter-mismatch
    and leak scenarios before being considered deployable.
    """

    candidates = [
        i for i, margin in enumerate(trace.safety) if margin >= minimum_observed_margin
    ]
    candidates.sort(key=lambda i: trace.cost[i])
    unique: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index in candidates:
        key = tuple(np.round(trace.gains[index], 12))
        if key not in seen:
            seen.add(key)
            unique.append(index)
        if len(unique) >= top_k:
            break
    if 0 not in unique:
        unique.append(0)

    accepted: list[tuple[float, int]] = []
    qualification_records: dict[int, tuple[float, float]] = {}
    for index in unique:
        results = [
            plant.simulate(trace.gains[index], seed=7_000_000 + scenario * 103)
            for scenario in range(qualification_scenarios)
        ]
        safe_rate = float(np.mean([result.safe for result in results]))
        mean_cost = float(np.mean([result.cost for result in results]))
        qualification_records[index] = (mean_cost, safe_rate)
        if safe_rate == 1.0:
            accepted.append((mean_cost, index))

    selected = min(accepted)[1] if accepted else 0
    qualification_cost, qualification_safe_rate = qualification_records[selected]
    record: dict[str, object] = {
        "method": trace.method,
        "seed": trace.seed,
        "candidates_tested": len(unique),
        "candidates_accepted": len(accepted),
        "selected_trial": selected + 1,
        "qualification_cost": qualification_cost,
        "qualification_safe_rate": qualification_safe_rate,
    }
    for name, value in zip(("Kp1", "Ki1", "Kp2", "Ki2"), trace.gains[selected]):
        record[name] = value
    return trace.gains[selected], record


def _validation_rows(
    trace: SearchTrace,
    plant: CoupledTankPlant,
    validation_scenarios: int,
    gains: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for scenario in range(validation_scenarios):
        result = plant.simulate(gains, seed=9_000_000 + scenario * 101)
        row: dict[str, object] = {
            "method": trace.method,
            "seed": trace.seed,
            "scenario": scenario,
            "cost": result.cost,
            "iae": result.iae,
            "energy": result.energy,
            "total_variation": result.total_variation,
            "safety_margin": result.safety_margin,
            "safe": result.safe,
            "saturation_fraction": result.saturation_fraction,
            "max_height": float(np.max(result.height)),
        }
        for name, value in zip(("Kp1", "Ki1", "Kp2", "Ki2"), gains):
            row[name] = value
        rows.append(row)
    return rows


def _mean_ci(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby("trial")[column]
    mean = grouped.mean().to_numpy()
    count = grouped.count().to_numpy()
    sem = grouped.std(ddof=1).fillna(0.0).to_numpy() / np.sqrt(np.maximum(count, 1))
    trial = grouped.mean().index.to_numpy()
    return trial, mean, 1.96 * sem


def _paired_wilcoxon_p(
    left: pd.Series, right: pd.Series, alternative: str
) -> float:
    if len(left) < 2 or np.allclose(left.to_numpy(), right.to_numpy()):
        return float("nan")
    return float(wilcoxon(left, right, alternative=alternative).pvalue)


def _plot_learning_curves(trials: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in METHODS:
        frame = trials[trials.method == method]
        x, mean, ci = _mean_ci(frame, "best_safe_cost")
        axes[0].plot(x, mean, label=LABELS[method], color=COLORS[method], lw=2)
        axes[0].fill_between(x, mean - ci, mean + ci, color=COLORS[method], alpha=0.15)
        x, mean, ci = _mean_ci(frame, "cumulative_unsafe_trials")
        axes[1].plot(x, mean, label=LABELS[method], color=COLORS[method], lw=2)
        axes[1].fill_between(x, mean - ci, mean + ci, color=COLORS[method], alpha=0.15)

    axes[0].set(title="Best observed safe cost", xlabel="Online trials", ylabel="Cost (lower is better)")
    axes[1].set(title="Cumulative unsafe trials", xlabel="Online trials", ylabel="Count")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "learning_and_safety.png", dpi=180)
    plt.close(fig)


def _plot_validation(validation: pd.DataFrame, output: Path) -> None:
    summary = validation.groupby("method").agg(
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
        safe_rate=("safe", "mean"),
    ).reindex(METHODS)
    labels = [LABELS[m] for m in METHODS]
    colors = [COLORS[m] for m in METHODS]
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x, summary.cost_mean, yerr=summary.cost_std, color=colors, capsize=4)
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel("Validation cost")
    axes[0].set_title("Robust performance")
    axes[1].bar(x, 100.0 * summary.safe_rate, color=colors)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Safe validation rollouts (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Out-of-sample safety")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "validation_summary.png", dpi=180)
    plt.close(fig)


def _plot_safety_calibration(trials: pd.DataFrame, output: Path) -> None:
    frame = trials[
        (trials.method == "safe_bo")
        & trials.predicted_safety_lcb.notna()
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    safe = frame.safe.astype(bool)
    ax.scatter(
        frame.loc[safe, "predicted_safety_lcb"],
        frame.loc[safe, "safety_margin"],
        s=22,
        alpha=0.55,
        color="#059669",
        label="Actually safe",
    )
    ax.scatter(
        frame.loc[~safe, "predicted_safety_lcb"],
        frame.loc[~safe, "safety_margin"],
        s=28,
        alpha=0.7,
        color="#dc2626",
        label="Unexpected violation",
    )
    ax.axhline(0.0, color="black", lw=1.1, ls="--")
    ax.axvline(0.0, color="black", lw=1.1, ls="--")
    ax.set(
        title="Safety confidence calibration",
        xlabel="Predicted safety lower confidence bound",
        ylabel="Observed safety margin",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "safety_calibration.png", dpi=180)
    plt.close(fig)


def _plot_representative(
    deployment_gains: dict[str, np.ndarray],
    plant: CoupledTankPlant,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for method in METHODS:
        result = plant.simulate(deployment_gains[method], seed=9_999_999)
        for loop in (0, 1):
            axes[0, loop].plot(
                result.time,
                result.height[:, loop],
                color=COLORS[method],
                label=LABELS[method],
                lw=1.7,
            )
            axes[1, loop].plot(
                result.time,
                result.command[:, loop],
                color=COLORS[method],
                label=LABELS[method],
                lw=1.4,
            )
    nominal = plant.simulate(deployment_gains["manual"], seed=9_999_999)
    for loop in (0, 1):
        axes[0, loop].plot(nominal.time, nominal.reference[:, loop], "k--", lw=1.2, label="Reference")
        axes[0, loop].axhline(plant.parameters.safe_max, color="#991b1b", ls=":", lw=1.3)
        axes[0, loop].set_ylabel(f"Tank {loop + 1} level (m)")
        axes[1, loop].set_ylabel(f"Pump {loop + 1} command")
        axes[1, loop].set_xlabel("Time (s)")
        axes[0, loop].grid(alpha=0.2)
        axes[1, loop].grid(alpha=0.2)
    axes[0, 0].legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "representative_response.png", dpi=180)
    plt.close(fig)


def run_experiment(
    output: Path,
    seeds: int,
    qualification_scenarios: int,
    validation_scenarios: int,
    config: SearchConfig,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    plant = CoupledTankPlant()
    space = ControllerSpace()
    trial_rows: list[dict[str, object]] = []
    qualification_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    representative: dict[str, np.ndarray] = {}

    for seed in range(seeds):
        for method in METHODS:
            trace = run_search(method, plant, space, seed=seed + 1, config=config)
            trial_rows.extend(_trace_rows(trace))
            gains, qualification = _qualify_controller(
                trace, plant, qualification_scenarios
            )
            qualification_rows.append(qualification)
            validation_rows.extend(
                _validation_rows(trace, plant, validation_scenarios, gains)
            )
            if seed == 0:
                representative[method] = gains
            print(
                f"seed={seed + 1:02d} method={method:8s} "
                f"unsafe={sum(not x for x in trace.safe):2d} "
                f"best_safe_cost={trace.cost[trace.best_safe_index()]:.4f}",
                flush=True,
            )

    trials = pd.DataFrame(trial_rows)
    qualifications = pd.DataFrame(qualification_rows)
    validation = pd.DataFrame(validation_rows)
    trials.to_csv(output / "trials.csv", index=False)
    qualifications.to_csv(output / "qualified_controllers.csv", index=False)
    validation.to_csv(output / "validation.csv", index=False)

    aggregate = validation.groupby("method").agg(
        validation_cost_mean=("cost", "mean"),
        validation_cost_std=("cost", "std"),
        validation_safe_rate=("safe", "mean"),
        validation_margin_mean=("safety_margin", "mean"),
    )
    search = trials.groupby("method").agg(
        online_unsafe_trials_mean=("safe", lambda x: float((~x.astype(bool)).sum()) / seeds),
        final_best_safe_cost_mean=("best_safe_cost", lambda x: float(x.groupby(trials.loc[x.index, "seed"]).last().mean())),
    )
    summary_frame = aggregate.join(search).reindex(METHODS)
    summary_frame.to_csv(output / "summary.csv")

    _plot_learning_curves(trials, output)
    _plot_validation(validation, output)
    _plot_safety_calibration(trials, output)
    _plot_representative(representative, plant, output)

    online_per_run = trials.groupby(["method", "seed"]).agg(
        unsafe_trials=("safe", lambda x: int((~x.astype(bool)).sum())),
        final_best_safe_cost=("best_safe_cost", "last"),
    ).reset_index()
    unsafe_wide = online_per_run.pivot(index="seed", columns="method", values="unsafe_trials")
    cost_wide = online_per_run.pivot(index="seed", columns="method", values="final_best_safe_cost")
    calibration = trials[
        (trials.method == "safe_bo") & trials.predicted_safety_lcb.notna()
    ]
    bo_unsafe_mean = float(unsafe_wide.bo.mean())
    violation_reduction = (
        100.0 * (1.0 - float(unsafe_wide.safe_bo.mean()) / bo_unsafe_mean)
        if bo_unsafe_mean > 0.0
        else float("nan")
    )
    derived = {
        "online_violation_reduction_vs_bo_percent": violation_reduction,
        "online_cost_penalty_vs_bo_percent": float(
            100.0 * (cost_wide.safe_bo.mean() / cost_wide.bo.mean() - 1.0)
        ),
        "paired_wilcoxon_unsafe_less_p": _paired_wilcoxon_p(
            unsafe_wide.safe_bo, unsafe_wide.bo, "less"
        ),
        "paired_wilcoxon_cost_greater_p": _paired_wilcoxon_p(
            cost_wide.safe_bo, cost_wide.bo, "greater"
        ),
        "certified_safe_bo_trials": int(len(calibration)),
        "certified_trials_with_observed_violation": int(
            (~calibration.safe.astype(bool)).sum()
        ),
    }

    summary: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seeds": seeds,
        "qualification_scenarios_per_candidate": qualification_scenarios,
        "validation_scenarios_per_run": validation_scenarios,
        "search_config": asdict(config),
        "plant_parameters": asdict(plant.parameters),
        "simulation_config": asdict(plant.config),
        "methods": summary_frame.reset_index().to_dict(orient="records"),
        "derived_comparisons": derived,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/final"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--qualification-scenarios", type=int, default=5)
    parser.add_argument("--validation-scenarios", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = 3
        args.budget = 18
        args.candidates = 768
        args.qualification_scenarios = 3
        args.validation_scenarios = 8
        if args.output == Path("results/final"):
            args.output = Path("results/quick")
    config = SearchConfig(budget=args.budget, num_candidates=args.candidates)
    run_experiment(
        args.output,
        args.seeds,
        args.qualification_scenarios,
        args.validation_scenarios,
        config,
    )


if __name__ == "__main__":
    main()
