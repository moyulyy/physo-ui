"""Matplotlib helpers (embedded canvases and export figures)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from . import theme

INK = theme.INK
GRAY = theme.GRAY
BLUE = theme.BLUE
GREEN = theme.GREEN
ORANGE = theme.ORANGE
GRID = "#ECECF1"


def _rc() -> None:
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Segoe UI", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 9


def new_figure(width: float = 5.2, height: float = 3.4, dpi: int = 110) -> Figure:
    _rc()
    fig = Figure(figsize=(width, height), dpi=dpi, facecolor="white")
    return fig


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 5.2, height: float = 3.4, parent=None):
        self.fig = new_figure(width, height)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background: transparent;")
        self.setMinimumHeight(220)

    def clear_message(self, message: str = "暂无数据") -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", color=theme.GRAY, fontsize=10)
        self.draw_idle()


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D9D9E0")
    ax.tick_params(colors=GRAY, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_history(canvas: PlotCanvas, history: Iterable[dict]) -> None:
    rows = [h for h in history if h.get("epoch") is not None]
    fig = canvas.fig
    fig.clear()
    ax = fig.add_subplot(111)
    _style(ax)
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "等待第一个 epoch…", ha="center", va="center", color=GRAY, fontsize=10)
        canvas.draw_idle()
        return
    epochs = [int(r["epoch"]) for r in rows]
    best = [r.get("best_r2", np.nan) for r in rows]
    mean = [r.get("mean_r2", np.nan) for r in rows]
    ax.plot(epochs, best, color=BLUE, linewidth=2.0, label="最佳 R²")
    ax.plot(epochs, mean, color=GRAY, linewidth=1.0, alpha=0.75, label="本批平均 R²")
    ax.set_ylim(-0.15, 1.02)
    ax.set_xlabel("Epoch", color=GRAY)
    ax.set_ylabel("R²", color=GRAY)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.18)
    canvas.draw_idle()


def draw_parity(canvas: PlotCanvas, y_true: np.ndarray, y_pred: np.ndarray, title: str = "预测 vs 真实") -> None:
    fig = canvas.fig
    fig.clear()
    ax = fig.add_subplot(111)
    _style(ax)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "没有可绘制的有效点", ha="center", va="center", color=GRAY)
        canvas.draw_idle()
        return
    t, p = y_true[mask], y_pred[mask]
    ax.scatter(t, p, s=13, color=BLUE, alpha=0.55, edgecolors="none")
    lo = float(min(t.min(), p.min()))
    hi = float(max(t.max(), p.max()))
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=ORANGE, linewidth=1.2, linestyle="--", label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("真实值", color=GRAY)
    ax.set_ylabel("预测值", color=GRAY)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.90, bottom=0.16)
    canvas.draw_idle()


def draw_residuals(canvas: PlotCanvas, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig = canvas.fig
    fig.clear()
    ax = fig.add_subplot(111)
    _style(ax)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "没有可绘制的有效点", ha="center", va="center", color=GRAY)
        canvas.draw_idle()
        return
    residual = y_true[mask] - y_pred[mask]
    ax.scatter(y_pred[mask], residual, s=13, color=GREEN, alpha=0.55, edgecolors="none")
    ax.axhline(0, color=ORANGE, linewidth=1.1, linestyle="--")
    ax.set_xlabel("预测值", color=GRAY)
    ax.set_ylabel("残差", color=GRAY)
    ax.set_title("残差分布", color=INK, fontsize=10, loc="left")
    fig.subplots_adjust(left=0.13, right=0.96, top=0.90, bottom=0.16)
    canvas.draw_idle()


def draw_scatter(canvas: PlotCanvas, x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> None:
    fig = canvas.fig
    fig.clear()
    ax = fig.add_subplot(111)
    _style(ax)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=12, color=BLUE, alpha=0.55, edgecolors="none")
    ax.set_xlabel(x_name, color=GRAY)
    ax.set_ylabel(y_name, color=GRAY)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.22)
    canvas.draw_idle()


def draw_pareto(canvas: PlotCanvas, candidates: list) -> None:
    fig = canvas.fig
    fig.clear()
    ax = fig.add_subplot(111)
    _style(ax)
    points = [(c.complexity, c.r2_train, c.expression) for c in candidates if np.isfinite(c.r2_train)]
    if not points:
        ax.axis("off")
        ax.text(0.5, 0.5, "暂无 Pareto 前沿", ha="center", va="center", color=GRAY)
        canvas.draw_idle()
        return
    complexities = [p[0] for p in points]
    r2s = [p[1] for p in points]
    ax.plot(complexities, r2s, color=BLUE, linewidth=1.2, marker="o", markersize=4, alpha=0.85)
    ax.set_xlabel("表达式复杂度", color=GRAY)
    ax.set_ylabel("训练 R²", color=GRAY)
    ax.set_title("精度 – 复杂度权衡", color=INK, fontsize=10, loc="left")
    ax.set_ylim(max(-0.1, min(r2s) - 0.05), 1.02)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.19)
    canvas.draw_idle()


def figure_parity(y_true: np.ndarray, y_pred: np.ndarray, title: str = "预测 vs 真实") -> Figure:
    fig = new_figure(5.4, 3.6)
    ax = fig.add_subplot(111)
    _style(ax)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum():
        t, p = y_true[mask], y_pred[mask]
        ax.scatter(t, p, s=13, color=BLUE, alpha=0.55, edgecolors="none")
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        pad = 0.06 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=ORANGE, linewidth=1.2, linestyle="--", label="y = x")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("真实值", color=GRAY)
    ax.set_ylabel("预测值", color=GRAY)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.90, bottom=0.16)
    return fig


def figure_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
    fig = new_figure(5.4, 3.6)
    ax = fig.add_subplot(111)
    _style(ax)
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum():
        ax.scatter(y_pred[mask], y_true[mask] - y_pred[mask], s=13, color=GREEN, alpha=0.55, edgecolors="none")
        ax.axhline(0, color=ORANGE, linewidth=1.1, linestyle="--")
    ax.set_xlabel("预测值", color=GRAY)
    ax.set_ylabel("残差", color=GRAY)
    ax.set_title("残差分布", color=INK, fontsize=10, loc="left")
    fig.subplots_adjust(left=0.13, right=0.96, top=0.90, bottom=0.16)
    return fig
