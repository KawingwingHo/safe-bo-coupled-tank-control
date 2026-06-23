"""Generate architecture artwork and a data-driven MP4 demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from .plant import CoupledTankPlant


FONT = "Heiti SC"
COLORS = {
    "bo": "#dc2626",
    "safe_bo": "#059669",
    "ink": "#172033",
    "muted": "#64748b",
    "water": "#38bdf8",
}


def _box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        0.17,
        0.105,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.5,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + 0.085, y + 0.0525, text, ha="center", va="center", fontsize=12, color=COLORS["ink"])


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#475569") -> None:
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, lw=1.5, color=color)
    )


def generate_architecture(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = FONT
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.94, "安全贝叶斯优化多回路 PI 自动整定架构", ha="center", fontsize=22, weight="bold", color=COLORS["ink"])

    _box(ax, (0.06, 0.57), "目标液位", "#334155")
    _box(ax, (0.27, 0.57), "双回路 PI", "#2563eb")
    _box(ax, (0.48, 0.57), "泵与执行器滞后", "#2563eb")
    _box(ax, (0.69, 0.57), "耦合双水箱", "#0284c7")
    _arrow(ax, (0.23, 0.622), (0.27, 0.622))
    _arrow(ax, (0.44, 0.622), (0.48, 0.622))
    _arrow(ax, (0.65, 0.622), (0.69, 0.622))

    _box(ax, (0.69, 0.33), "液位与泵状态", "#0284c7")
    _arrow(ax, (0.775, 0.57), (0.775, 0.435))
    _arrow(ax, (0.69, 0.382), (0.355, 0.57))
    ax.text(0.51, 0.43, "传感器反馈", ha="center", fontsize=11, color=COLORS["muted"])

    _box(ax, (0.06, 0.20), "性能 GP", "#7c3aed")
    _box(ax, (0.27, 0.20), "安全 GP + LCB", "#7c3aed")
    _box(ax, (0.48, 0.20), "0.18 信任步长", "#059669")
    _box(ax, (0.69, 0.20), "5 工况资格门", "#059669")
    _arrow(ax, (0.23, 0.252), (0.27, 0.252))
    _arrow(ax, (0.44, 0.252), (0.48, 0.252))
    _arrow(ax, (0.65, 0.252), (0.69, 0.252))
    _arrow(ax, (0.355, 0.305), (0.355, 0.57), color="#7c3aed")
    ax.text(0.37, 0.47, "候选 PI 参数", fontsize=11, color="#7c3aed", rotation=90, va="center")
    _arrow(ax, (0.775, 0.33), (0.775, 0.305), color="#7c3aed")

    ax.text(0.5, 0.08, "在线阶段控制试验风险，离线资格门控制部署风险；二者不能互相替代", ha="center", fontsize=12, color=COLORS["muted"])
    fig.tight_layout()
    path = output / "system_architecture.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _tank(ax: plt.Axes, x: float, level: float, title: str, color: str, safe_max: float) -> None:
    width, height = 0.28, 0.58
    y = 0.18
    ax.add_patch(Rectangle((x, y), width, height, fill=False, ec="#334155", lw=2.0))
    water_height = height * np.clip(level / 0.45, 0.0, 1.0)
    ax.add_patch(Rectangle((x, y), width, water_height, color=COLORS["water"], alpha=0.75))
    safe_y = y + height * safe_max / 0.45
    ax.plot([x - 0.02, x + width + 0.02], [safe_y, safe_y], color="#b91c1c", ls="--", lw=1.5)
    ax.text(x + width / 2, y + height + 0.06, title, ha="center", fontsize=12, weight="bold", color=color)
    ax.text(x + width / 2, y + water_height / 2, f"{level:.3f} m", ha="center", va="center", fontsize=11, color="#0c4a6e")


def _gains_from_row(row: pd.Series) -> np.ndarray:
    return row[["Kp1", "Ki1", "Kp2", "Ki2"]].to_numpy(dtype=float)


def generate_video(results: Path, output: Path, fps: int = 20, seconds: int = 45) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = FONT
    trials = pd.read_csv(results / "trials.csv")
    qualified = pd.read_csv(results / "qualified_controllers.csv")
    summary = pd.read_csv(results / "summary.csv").set_index("method")
    plant = CoupledTankPlant()

    risky = trials[(trials.method == "bo") & (trials.seed == 1)].sort_values("safety_margin").iloc[0]
    safe_row = qualified[(qualified.method == "safe_bo") & (qualified.seed == 1)].iloc[0]
    risky_result = plant.simulate(_gains_from_row(risky), seed=100_017)
    safe_result = plant.simulate(_gains_from_row(safe_row), seed=100_017)

    learning = trials.groupby(["method", "trial"]).cumulative_unsafe_trials.mean().unstack(0)
    total_frames = fps * seconds
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#f8fafc")
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    def update(frame: int) -> None:
        fig.clear()
        elapsed = frame / fps
        if elapsed < 4.0:
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.66, "安全贝叶斯优化自动整定", ha="center", fontsize=34, weight="bold", color=COLORS["ink"])
            ax.text(0.5, 0.53, "耦合双水箱多回路 PI 软件在环实验", ha="center", fontsize=22, color="#334155")
            ax.text(0.5, 0.36, "问题：如何在减少调参试验风险的同时保持控制性能？", ha="center", fontsize=17, color=COLORS["muted"])
            return

        if elapsed < 27.0:
            progress = (elapsed - 4.0) / 23.0
            index = min(int(progress * (len(risky_result.time) - 1)), len(risky_result.time) - 1)
            grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], hspace=0.28, wspace=0.20)
            for column, (title, result, color) in enumerate(
                [
                    ("普通 BO 在线危险试验", risky_result, COLORS["bo"]),
                    ("安全 BO 认证控制器", safe_result, COLORS["safe_bo"]),
                ]
            ):
                tank_ax = fig.add_subplot(grid[0, column])
                tank_ax.set_xlim(0, 1)
                tank_ax.set_ylim(0, 1)
                tank_ax.axis("off")
                _tank(tank_ax, 0.16, result.height[index, 0], "Tank 1", color, plant.parameters.safe_max)
                _tank(tank_ax, 0.56, result.height[index, 1], "Tank 2", color, plant.parameters.safe_max)
                saturated = (result.command[: index + 1] >= 0.995) | (
                    result.command[: index + 1] <= 0.005
                )
                saturation = 100.0 * float(np.mean(saturated))
                final_status = "SAFE" if result.safe else "UNSAFE"
                tank_ax.text(
                    0.5,
                    0.04,
                    f"t={result.time[index]:5.1f}s   累计泵饱和={saturation:4.1f}%   最终判定={final_status}",
                    ha="center",
                    fontsize=12,
                    color=("#047857" if result.safe else "#b91c1c"),
                    weight="bold",
                )

                curve_ax = fig.add_subplot(grid[1, column])
                curve_ax.plot(result.time[: index + 1], result.height[: index + 1, 0], color=color, lw=2, label="Tank 1")
                curve_ax.plot(result.time[: index + 1], result.height[: index + 1, 1], color=color, lw=1.5, alpha=0.65, label="Tank 2")
                curve_ax.plot(result.time[: index + 1], result.reference[: index + 1, 0], "k--", lw=1, label="Reference")
                curve_ax.axhline(plant.parameters.safe_max, color="#991b1b", ls=":", lw=1.4, label="Safety limit")
                curve_ax.set(xlim=(0, plant.config.duration), ylim=(0.10, 0.39), xlabel="Time (s)", ylabel="Level (m)", title=title)
                curve_ax.grid(alpha=0.2)
                if column == 0:
                    curve_ax.legend(frameon=False, ncol=2, fontsize=8)
            fig.suptitle("同一 commissioning profile 下的参数试验", fontsize=19, weight="bold", color=COLORS["ink"])
            return

        if elapsed < 38.0:
            progress = (elapsed - 27.0) / 11.0
            upto = max(1, min(int(progress * 30), 30))
            ax = fig.add_subplot(111)
            ax.plot(learning.index[:upto], learning.bo.iloc[:upto], color=COLORS["bo"], lw=3, label="Ordinary BO")
            ax.plot(learning.index[:upto], learning.safe_bo.iloc[:upto], color=COLORS["safe_bo"], lw=3, label="Safe BO")
            ax.fill_between(learning.index[:upto], learning.safe_bo.iloc[:upto], color=COLORS["safe_bo"], alpha=0.10)
            ax.set(xlim=(1, 30), ylim=(0, 21), xlabel="Online trials", ylabel="Mean cumulative unsafe trials", title="安全域约束改变的是调参过程，而不只是最终参数")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=13)
            ax.text(0.02, 0.90, "普通 BO：19.05 次/工况", transform=ax.transAxes, fontsize=15, color=COLORS["bo"], weight="bold")
            ax.text(0.02, 0.83, "安全 BO：0.90 次/工况", transform=ax.transAxes, fontsize=15, color=COLORS["safe_bo"], weight="bold")
            fig.tight_layout(pad=3)
            return

        ax = fig.add_subplot(111)
        ax.axis("off")
        safe = summary.loc["safe_bo"]
        bo = summary.loc["bo"]
        metrics = [
            ("在线违规减少", "95.28%"),
            ("安全 BO 未见工况安全率", f"{100 * safe.validation_safe_rate:.2f}%"),
            ("安全 BO 验证成本", f"{safe.validation_cost_mean:.4f}"),
            ("普通 BO 验证成本", f"{bo.validation_cost_mean:.4f}"),
        ]
        ax.text(0.5, 0.82, "最终结果", ha="center", fontsize=30, weight="bold", color=COLORS["ink"])
        for i, (name, value) in enumerate(metrics):
            y = 0.66 - i * 0.12
            ax.text(0.30, y, name, ha="right", fontsize=17, color="#475569")
            ax.text(0.34, y, value, ha="left", fontsize=20, color=COLORS["safe_bo"], weight="bold")
        ax.text(0.5, 0.12, "结论：显著降低在线探索风险，但 GP 高概率安全不等于绝对安全", ha="center", fontsize=16, color="#b45309")

    movie = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps)
    path = output / "safe_bo_coupled_tank_demo.mp4"
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2800, extra_args=["-pix_fmt", "yuv420p"])
    movie.save(path, writer=writer)
    update(int(39 * fps))
    fig.savefig(output / "demo_preview.png", dpi=160, facecolor="#f8fafc")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/final"))
    parser.add_argument("--output", type=Path, default=Path("demo"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    generate_architecture(Path("assets"))
    if not args.assets_only:
        generate_video(args.results, args.output, args.fps, args.seconds)


if __name__ == "__main__":
    main()
