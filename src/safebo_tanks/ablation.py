"""Reproduce the trust-region safety ablation used in the engineering audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .optimization import SearchConfig, run_search
from .plant import CoupledTankPlant
from .space import ControllerSpace


VARIANTS = {
    "lcb_only": {"max_safe_step": 2.0},
    "trust_region": {"max_safe_step": 0.18},
}


def _paired_p(left: pd.Series, right: pd.Series, alternative: str) -> float:
    if len(left) < 2 or np.allclose(left.to_numpy(), right.to_numpy()):
        return float("nan")
    return float(wilcoxon(left, right, alternative=alternative).pvalue)


def run_ablation(
    output: Path,
    seeds: int = 20,
    budget: int = 30,
    candidates: int = 2048,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    plant = CoupledTankPlant()
    space = ControllerSpace()
    base = SearchConfig(budget=budget, num_candidates=candidates)
    rows: list[dict[str, object]] = []

    for variant, overrides in VARIANTS.items():
        config = replace(base, **overrides)
        for seed in range(1, seeds + 1):
            trace = run_search("safe_bo", plant, space, seed, config)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "unsafe_trials": sum(not value for value in trace.safe),
                    "best_safe_cost": trace.cost[trace.best_safe_index()],
                    "max_safe_step": config.max_safe_step,
                }
            )
            print(
                f"variant={variant:12s} seed={seed:02d} "
                f"unsafe={rows[-1]['unsafe_trials']:2d} "
                f"cost={rows[-1]['best_safe_cost']:.4f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "trust_region_ablation.csv", index=False)
    aggregate = frame.groupby("variant").agg(
        unsafe_mean=("unsafe_trials", "mean"),
        unsafe_std=("unsafe_trials", "std"),
        cost_mean=("best_safe_cost", "mean"),
        cost_std=("best_safe_cost", "std"),
    ).reindex(VARIANTS)

    wide_unsafe = frame.pivot(index="seed", columns="variant", values="unsafe_trials")
    wide_cost = frame.pivot(index="seed", columns="variant", values="best_safe_cost")
    unsafe_p = _paired_p(
        wide_unsafe.trust_region,
        wide_unsafe.lcb_only,
        "less",
    )
    cost_p = _paired_p(
        wide_cost.trust_region,
        wide_cost.lcb_only,
        "two-sided",
    )

    labels = ["LCB only", "LCB + trust region"]
    colors = ["#ef4444", "#059669"]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    unsafe_ci = 1.96 * aggregate.unsafe_std / np.sqrt(seeds)
    cost_ci = 1.96 * aggregate.cost_std / np.sqrt(seeds)
    axes[0].bar(
        x,
        aggregate.unsafe_mean,
        yerr=unsafe_ci,
        color=colors,
        capsize=4,
    )
    axes[0].set(ylabel="Unsafe online trials", title="Safety ablation")
    axes[1].bar(
        x,
        aggregate.cost_mean,
        yerr=cost_ci,
        color=colors,
        capsize=4,
    )
    axes[1].set(ylabel="Best safe cost", title="Performance trade-off")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=10)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(output / "trust_region_ablation.png", dpi=180)
    plt.close(fig)

    summary: dict[str, object] = {
        "seeds": seeds,
        "base_config": asdict(base),
        "variants": aggregate.reset_index().to_dict(orient="records"),
        "paired_wilcoxon_trust_region_fewer_violations_p": unsafe_p,
        "paired_wilcoxon_cost_difference_p": cost_p,
    }
    (output / "trust_region_ablation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/ablation"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds = 3
        args.budget = 12
        args.candidates = 512
    run_ablation(args.output, args.seeds, args.budget, args.candidates)


if __name__ == "__main__":
    main()
