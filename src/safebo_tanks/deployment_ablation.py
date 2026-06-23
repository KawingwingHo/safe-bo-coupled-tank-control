"""Reproduce the robust deployment-gate ablation for Safe BO controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .plant import CoupledTankPlant


GAIN_COLUMNS = ["Kp1", "Ki1", "Kp2", "Ki2"]
VARIANTS = ("direct_best_safe", "qualification_gate")


def _paired_p(left: pd.Series, right: pd.Series, alternative: str) -> float:
    if len(left) < 2 or np.allclose(left.to_numpy(), right.to_numpy()):
        return float("nan")
    return float(wilcoxon(left, right, alternative=alternative).pvalue)


def _controller_map(
    trials: pd.DataFrame, qualified: pd.DataFrame
) -> dict[tuple[str, int], np.ndarray]:
    safe_bo = trials[(trials.method == "safe_bo") & trials.safe.astype(bool)]
    direct = (
        safe_bo.sort_values(["seed", "cost"])
        .groupby("seed", as_index=False)
        .first()
    )
    qualified_safe_bo = qualified[qualified.method == "safe_bo"]
    controllers: dict[tuple[str, int], np.ndarray] = {}
    for _, row in direct.iterrows():
        controllers[("direct_best_safe", int(row.seed))] = row[GAIN_COLUMNS].to_numpy(
            dtype=float
        )
    for _, row in qualified_safe_bo.iterrows():
        controllers[("qualification_gate", int(row.seed))] = row[
            GAIN_COLUMNS
        ].to_numpy(dtype=float)
    return controllers


def run_deployment_ablation(
    trials_path: Path,
    qualified_path: Path,
    output: Path,
    validation_scenarios: int = 20,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(trials_path)
    qualified = pd.read_csv(qualified_path)
    controllers = _controller_map(trials, qualified)
    seeds = sorted({seed for variant, seed in controllers if variant == VARIANTS[0]})
    if not seeds:
        raise ValueError("no Safe BO controllers found")
    for seed in seeds:
        if any((variant, seed) not in controllers for variant in VARIANTS):
            raise ValueError(f"missing deployment controller for seed {seed}")

    plant = CoupledTankPlant()
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for seed in seeds:
            gains = controllers[(variant, seed)]
            for scenario in range(validation_scenarios):
                result = plant.simulate(
                    gains, seed=9_000_000 + scenario * 101
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "scenario": scenario,
                        "cost": result.cost,
                        "safe": result.safe,
                        "safety_margin": result.safety_margin,
                        **dict(zip(GAIN_COLUMNS, gains, strict=True)),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "deployment_gate_ablation.csv", index=False)
    per_seed = frame.groupby(["variant", "seed"]).agg(
        safe_rate=("safe", "mean"),
        mean_cost=("cost", "mean"),
        mean_margin=("safety_margin", "mean"),
    )
    aggregate = per_seed.groupby("variant").agg(
        safe_rate_mean=("safe_rate", "mean"),
        safe_rate_std=("safe_rate", "std"),
        cost_mean=("mean_cost", "mean"),
        cost_std=("mean_cost", "std"),
        margin_mean=("mean_margin", "mean"),
    ).reindex(VARIANTS)

    wide_safety = per_seed.safe_rate.unstack("variant")
    wide_cost = per_seed.mean_cost.unstack("variant")
    safety_p = _paired_p(
        wide_safety.qualification_gate,
        wide_safety.direct_best_safe,
        "greater",
    )
    cost_p = _paired_p(
        wide_cost.qualification_gate,
        wide_cost.direct_best_safe,
        "two-sided",
    )

    labels = ["Direct best-safe", "5-scenario gate"]
    colors = ["#dc2626", "#059669"]
    x = np.arange(2)
    count = len(seeds)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    safety_ci = 1.96 * aggregate.safe_rate_std / np.sqrt(count)
    cost_ci = 1.96 * aggregate.cost_std / np.sqrt(count)
    axes[0].bar(
        x,
        100.0 * aggregate.safe_rate_mean,
        yerr=100.0 * safety_ci,
        color=colors,
        capsize=4,
    )
    axes[0].set(ylabel="Safe held-out rollouts (%)", title="Deployment safety")
    axes[0].set_ylim(0.0, 105.0)
    axes[1].bar(x, aggregate.cost_mean, yerr=cost_ci, color=colors, capsize=4)
    axes[1].set(ylabel="Held-out cost", title="Deployment performance")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=8)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "deployment_gate_ablation.png", dpi=180)
    plt.close(fig)

    summary: dict[str, object] = {
        "seeds": len(seeds),
        "validation_scenarios_per_seed": validation_scenarios,
        "variants": aggregate.reset_index().to_dict(orient="records"),
        "paired_wilcoxon_gate_safer_p": safety_p,
        "paired_wilcoxon_cost_difference_p": cost_p,
    }
    (output / "deployment_gate_ablation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=Path("results/final/trials.csv"))
    parser.add_argument(
        "--qualified",
        type=Path,
        default=Path("results/final/qualified_controllers.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/ablation"))
    parser.add_argument("--validation-scenarios", type=int, default=20)
    args = parser.parse_args()
    run_deployment_ablation(
        args.trials, args.qualified, args.output, args.validation_scenarios
    )


if __name__ == "__main__":
    main()
