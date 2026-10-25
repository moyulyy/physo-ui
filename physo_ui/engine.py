"""Training engine around PhySO, inference helpers and result persistence."""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
import platform
import re
import shutil
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from . import data as D
from . import plotting

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
RESULT_SCHEMA = "physo-ui/result@1"

DEFAULT_OPS = [
    "mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos",
]
ALL_OPS = [
    "add", "sub", "mul", "div", "inv", "n2", "n3", "n4", "sqrt", "cbrt", "neg", "abs",
    "exp", "expneg", "log", "logabs", "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "arcsin", "arccos", "arctan", "erf", "pow",
]

PRESETS = {
    "config0": {"name": "快速", "note": "轻量配置，适合演示与快速验证。", "batch": 500},
    "config2": {"name": "平衡", "note": "Class 基准配置，速度与效果折中。", "batch": 1000},
    "config1": {"name": "高精度", "note": "论文级配置，搜索更充分，耗时也明显更长。", "batch": 2000},
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    preset: str = "config0"
    epochs: int = 10
    batch_size: int = 500
    lr: float = 0.0025
    max_length: int = 20
    stop_reward: float = 0.9999
    stop_after_n_epochs: int = 2
    max_n_evaluations: int | None = None
    seed: int = 0
    device: str = "cpu"
    parallel: bool = False
    ops: list[str] = field(default_factory=lambda: list(DEFAULT_OPS))
    n_free_consts: int = 2
    fixed_consts: list[float] = field(default_factory=lambda: [1.0, math.pi])
    risk_factor: float = 0.05
    entropy_weight: float = 0.005
    n_cpus: int = 1
    cpu_threads: int = 0
    const_opti_steps: int = 6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in (payload or {}).items() if k in known}
        if "ops" in data:
            data["ops"] = [op for op in data["ops"] if op in ALL_OPS] or list(DEFAULT_OPS)
        try:
            return cls(**data)
        except TypeError:
            return cls()

    def resolve_device(self) -> str:
        if self.device in {"cpu", "cuda"}:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def environment_info() -> dict[str, Any]:
    from importlib import metadata

    def version(name: str) -> str:
        try:
            return metadata.version(name)
        except Exception:
            return "未安装"

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "physo": version("physo"),
        "torch": version("torch"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "sympy": version("sympy"),
        "pyside6": version("PySide6"),
        "cuda": "未知",
        "gpu": "—",
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda"] = str(torch.version.cuda or "可用")
            info["gpu"] = torch.cuda.get_device_name(0)
        else:
            info["cuda"] = "CPU only"
    except Exception:
        info["cuda"] = "不可用"
    return info


def make_run_dir(dataset_name: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", dataset_name).strip("_")[:40] or "run"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = RUNS_DIR / f"{stamp}-{safe}"
    suffix = 2
    while candidate.exists():
        candidate = RUNS_DIR / f"{stamp}-{safe}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


# --------------------------------------------------------------------------- #
# Result objects
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    expression: str
    display_expression: str
    pretty: str
    latex: str
    complexity: int = 0
    length: int = 0
    reward: float = float("nan")
    r2_train: float = float("nan")
    rmse_train: float = float("nan")
    r2_eval: float = float("nan")
    rmse_eval: float = float("nan")
    source: str = "pareto"
    free_consts: dict[str, float] = field(default_factory=dict)

    def score_text(self) -> str:
        return D.format_number(self.r2_train, 5)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Candidate":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class RunResult:
    summary: dict[str, Any]
    config: dict[str, Any]
    candidates: list[Candidate]
    active_index: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    run_dir: str = ""
    created_at: str = ""
    duration: float = 0.0
    epochs_done: int = 0
    stopped: bool = False
    data: D.PreparedData | None = None
    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    X_eval: np.ndarray | None = None
    y_eval: np.ndarray | None = None
    version: str = RESULT_SCHEMA

    # -- convenience -------------------------------------------------------- #
    @property
    def active(self) -> Candidate:
        if not self.candidates:
            raise ValueError("结果中没有候选表达式。")
        idx = max(0, min(int(self.active_index), len(self.candidates) - 1))
        return self.candidates[idx]

    @property
    def x_names(self) -> list[str]:
        return list(self.summary.get("x_names") or [])

    @property
    def x_columns(self) -> list[str]:
        return list(self.summary.get("x_columns") or self.x_names)

    @property
    def y_name(self) -> str:
        return str(self.summary.get("y_name") or "y")

    @property
    def y_column(self) -> str:
        return str(self.summary.get("y_column") or self.y_name)

    @property
    def normalize(self) -> bool:
        return bool(self.summary.get("normalize", False))

    @property
    def x_units(self) -> list[list[float]]:
        units = self.summary.get("x_units") or []
        if len(units) != len(self.x_names):
            units = [[0.0] * 7 for _ in self.x_names]
        return [[float(v) for v in vec] for vec in units]

    @property
    def y_units(self) -> list[float]:
        units = self.summary.get("y_units") or [0.0] * 7
        return [float(v) for v in units]

    @property
    def use_units(self) -> bool:
        if not bool(self.summary.get("use_units", True)):
            return False
        return any(abs(v) > 1e-12 for vec in self.x_units for v in vec) or any(
            abs(v) > 1e-12 for v in self.y_units
        )

    @property
    def is_loaded(self) -> bool:
        return self.data is None

    def activate(self, index: int) -> Candidate:
        if not self.candidates:
            raise ValueError("没有可用的候选表达式。")
        self.active_index = max(0, min(int(index), len(self.candidates) - 1))
        return self.active

    # -- prediction --------------------------------------------------------- #
    def predict_array(self, X_raw: np.ndarray) -> np.ndarray:
        X = np.asarray(X_raw, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1) if X.size == len(self.x_names) else X.reshape(1, -1)
        elif X.ndim == 2 and X.shape[0] != len(self.x_names) and X.shape[1] == len(self.x_names):
            X = X.T
        if self.normalize:
            mean = np.asarray(self.summary.get("x_mean", np.zeros(X.shape[0])), dtype=float).reshape(-1, 1)
            std = np.asarray(self.summary.get("x_std", np.ones(X.shape[0])), dtype=float).reshape(-1, 1)
            Xm = (X - mean[: X.shape[0]]) / std[: X.shape[0]]
        else:
            Xm = X
        pred = D.predict_expression(self.active.expression, self.x_names, Xm)
        if self.normalize:
            pred = pred * float(self.summary.get("y_std", 1.0)) + float(self.summary.get("y_mean", 0.0))
        return np.asarray(pred, dtype=float).ravel()

    def infer_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        columns: list[str] = []
        for orig, name in zip(self.x_columns, self.x_names):
            if orig in out.columns:
                columns.append(orig)
            elif name in out.columns:
                columns.append(name)
            else:
                raise ValueError(f"缺少输入列：{orig}（或 {name}）")
        X = out[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T
        pred = self.predict_array(X)
        out["prediction"] = pred
        y_col = self.y_column if self.y_column in out.columns else self.y_name
        if y_col in out.columns:
            truth = pd.to_numeric(out[y_col], errors="coerce").to_numpy(dtype=float)
            out["residual"] = truth - pred
        return out

    def prediction_frame(self) -> pd.DataFrame:
        if self.data is None or self.y_train is None:
            return pd.DataFrame()
        parts = []
        for split, X, y in (
            ("train", self.X_train, self.y_train),
            ("eval", self.X_eval, self.y_eval),
        ):
            if X is None or y is None or len(y) == 0:
                continue
            pred = self.predict_array(X)
            frame = pd.DataFrame({col: X[i] for i, col in enumerate(self.x_columns)})
            frame[self.y_column] = y
            frame["prediction"] = pred
            frame["residual"] = y - pred
            frame["split"] = split
            parts.append(frame)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "created_at": self.created_at,
            "duration": self.duration,
            "epochs_done": self.epochs_done,
            "stopped": self.stopped,
            "run_dir": self.run_dir,
            "summary": _jsonable(self.summary),
            "config": _jsonable(self.config),
            "candidates": [c.to_dict() for c in self.candidates],
            "active_index": self.active_index,
            "history": _jsonable(self.history),
            "environment": environment_info(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunResult":
        if payload.get("schema") not in (None, RESULT_SCHEMA):
            raise ValueError("结果文件版本不兼容。")
        candidates = [Candidate.from_dict(c) for c in payload.get("candidates", [])]
        if not candidates:
            raise ValueError("结果文件不包含候选表达式。")
        obj = cls(
            summary=payload.get("summary", {}),
            config=payload.get("config", {}),
            candidates=candidates,
            active_index=int(payload.get("active_index", 0)),
            history=payload.get("history", []),
            run_dir=payload.get("run_dir", ""),
            created_at=payload.get("created_at", ""),
            duration=float(payload.get("duration", 0.0)),
            epochs_done=int(payload.get("epochs_done", 0)),
            stopped=bool(payload.get("stopped", False)),
        )
        return obj

    def save_to_dir(self, target: str | Path) -> Path:
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / "result.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        active = self.active
        expression_lines = [
            "PhySO Studio — Symbolic Regression Result",
            "=" * 44,
            f"dataset      : {self.summary.get('name', '')}",
            f"created_at   : {self.created_at}",
            f"target       : {self.y_name}",
            f"inputs       : {', '.join(self.x_names)}",
            f"normalize    : {self.normalize}",
            f"R2 (train)   : {active.r2_train}",
            f"RMSE (train) : {active.rmse_train}",
            f"R2 (eval)    : {active.r2_eval}",
            f"expression   : {active.display_expression}",
            "",
            "Pretty form:",
            active.pretty,
            "",
            "LaTeX:",
            active.latex,
            "",
            "Model-space expression (used for inference):",
            active.expression,
        ]
        (target / "expression.txt").write_text("\n".join(expression_lines), encoding="utf-8")
        (target / "config.json").write_text(
            json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        predictions = self.prediction_frame()
        if not predictions.empty:
            predictions.to_csv(target / "predictions.csv", index=False, encoding="utf-8-sig")
        if self.data is not None:
            dataset = self.data.frame.copy()
            split = np.array(["train"] * len(dataset), dtype=object)
            if self.data.split_enabled and self.data.eval_idx.size:
                split[self.data.train_idx] = "train"
                split[self.data.eval_idx] = "eval"
            dataset["split"] = split
            dataset.to_csv(target / "dataset_used.csv", index=False, encoding="utf-8-sig")
        try:
            self._save_figures(target)
        except Exception:
            (target / "plot_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return target

    def _save_figures(self, target: Path) -> None:
        if self.history:
            fig = _figure_history(self.history)
            fig.savefig(target / "history.png", dpi=160, facecolor="white")
        if self.data is not None and self.y_train is not None:
            pred_train = self.predict_array(self.X_train)
            plotting.figure_parity(self.y_train, pred_train, "训练集：预测 vs 真实").savefig(
                target / "fit_train.png", dpi=160, facecolor="white"
            )
            plotting.figure_residuals(self.y_train, pred_train).savefig(
                target / "residuals_train.png", dpi=160, facecolor="white"
            )
        if self.data is not None and self.y_eval is not None and len(self.y_eval):
            pred_eval = self.predict_array(self.X_eval)
            plotting.figure_parity(self.y_eval, pred_eval, "评估集：预测 vs 真实").savefig(
                target / "fit_eval.png", dpi=160, facecolor="white"
            )
            plotting.figure_residuals(self.y_eval, pred_eval).savefig(
                target / "residuals_eval.png", dpi=160, facecolor="white"
            )
        if self.candidates:
            fig = _figure_pareto(self.candidates)
            fig.savefig(target / "pareto.png", dpi=160, facecolor="white")

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        path = Path(path)
        if path.is_dir():
            path = path / "result.json"
        if not path.exists():
            raise FileNotFoundError(f"找不到结果文件：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = cls.from_dict(payload)
        result._attach_dataset(path.parent)
        result._refresh_display_expressions()
        return result

    def _refresh_display_expressions(self) -> None:
        """Recompute readable display forms (also repairs results saved by older builds)."""
        for candidate in self.candidates:
            try:
                expr = D.expression_to_sympy(candidate.expression, self.x_names)
                display = expr
                if self.normalize:
                    physical = _physical_expression(
                        expr,
                        self.x_names,
                        np.asarray(self.summary.get("x_mean", []), dtype=float),
                        np.asarray(self.summary.get("x_std", []), dtype=float),
                        float(self.summary.get("y_mean", 0.0)),
                        float(self.summary.get("y_std", 1.0)),
                    )
                    if len(str(physical)) <= 1200:
                        display = physical
                from .mathrender import humanize_expression

                display = humanize_expression(_rounded_expr(display, 4))
                candidate.display_expression = str(display)
                candidate.pretty = safe_pretty(display)
                candidate.latex = safe_latex(display)
            except Exception:
                continue

    def _attach_dataset(self, folder: Path) -> None:
        """Re-attach the exported modelling data so loaded results stay fully usable."""
        dataset_path = folder / "dataset_used.csv"
        if not dataset_path.exists():
            return
        try:
            frame = pd.read_csv(dataset_path, encoding="utf-8-sig")
        except Exception:
            return
        x_cols = self.x_columns
        y_col = self.y_column
        if any(col not in frame.columns for col in [*x_cols, y_col]):
            return
        subset = frame[[*x_cols, y_col]].apply(pd.to_numeric, errors="coerce")
        valid = subset.notna().all(axis=1)
        subset = subset.loc[valid].reset_index(drop=True)
        if subset.empty:
            return
        X = subset[x_cols].to_numpy(dtype=float).T
        y = subset[y_col].to_numpy(dtype=float).ravel()
        n = len(subset)
        if "split" in frame.columns:
            split = frame.loc[valid, "split"].astype(str).to_numpy()
            train_idx = np.flatnonzero(split == "train")
            eval_idx = np.flatnonzero(split == "eval")
            split_enabled = eval_idx.size > 0
        else:
            train_idx = np.arange(n)
            eval_idx = np.array([], dtype=int)
            split_enabled = False
        if train_idx.size == 0:
            train_idx = np.arange(n)
        x_mean = np.asarray(self.summary.get("x_mean", [0.0] * len(x_cols)), dtype=float)
        x_std = np.asarray(self.summary.get("x_std", [1.0] * len(x_cols)), dtype=float)
        if x_mean.size != len(x_cols):
            x_mean = np.zeros(len(x_cols))
        if x_std.size != len(x_cols):
            x_std = np.ones(len(x_cols))
        data = D.PreparedData(
            name=str(self.summary.get("name", folder.name)),
            frame=subset,
            x_cols=x_cols,
            y_col=y_col,
            x_names=self.x_names,
            y_name=self.y_name,
            X_raw=X,
            y_raw=y,
            split_enabled=split_enabled,
            train_fraction=float(self.summary.get("train_fraction", 0.8)),
            normalize=self.normalize,
            train_idx=train_idx,
            eval_idx=eval_idx,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=float(self.summary.get("y_mean", 0.0)),
            y_std=float(self.summary.get("y_std", 1.0)),
            x_units=self.x_units,
            y_units=self.y_units,
            use_units=bool(self.summary.get("use_units", True)),
            source_path=str(dataset_path),
        )
        self.data = data
        self.X_train = X[:, train_idx]
        self.y_train = y[train_idx]
        self.X_eval = X[:, eval_idx] if eval_idx.size else np.empty((len(x_cols), 0))
        self.y_eval = y[eval_idx] if eval_idx.size else np.empty((0,))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _figure_history(history: list[dict[str, Any]]):
    history = [h for h in history if h.get("epoch") is not None]
    fig = plotting.new_figure(6.2, 3.8)
    ax = fig.add_subplot(111)
    epochs = [int(h["epoch"]) for h in history]
    ax.plot(epochs, [h.get("best_r2", np.nan) for h in history], color=plotting.BLUE, linewidth=2, label="best R²")
    ax.plot(epochs, [h.get("mean_r2", np.nan) for h in history], color=plotting.GRAY, linewidth=1, label="mean R²")
    ax.set_ylim(-0.15, 1.02)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("训练曲线", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="#ECECF1")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.16)
    return fig


def _figure_pareto(candidates: list[Candidate]):
    fig = plotting.new_figure(6.0, 3.6)
    ax = fig.add_subplot(111)
    points = [(c.complexity, c.r2_train) for c in candidates if np.isfinite(c.r2_train)]
    if points:
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", color=plotting.BLUE, linewidth=1.2)
        ax.set_ylim(max(-0.1, min(p[1] for p in points) - 0.05), 1.02)
    ax.set_xlabel("复杂度")
    ax.set_ylabel("训练 R²")
    ax.set_title("Pareto 前沿", loc="left")
    ax.grid(True, color="#ECECF1")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.17)
    return fig


# --------------------------------------------------------------------------- #
# Worker internals
# --------------------------------------------------------------------------- #
class TrainingCancelled(Exception):
    pass


class _NullVisualiser:
    def visualise(self, run_logger=None, batch=None):  # noqa: D102
        return None

    def save_visualisation(self):
        return None

    def save_data(self):
        return None

    def save_pareto_data(self):
        return None

    def save_pareto_fig(self):
        return None


class _CallbackLogger:
    """Subclass created lazily so importing this module does not require PhySO."""

    @staticmethod
    def build(monitoring, save_path: Path, on_epoch, cancel_event: threading.Event, on_log):
        class CallbackLogger(monitoring.RunLogger):
            def __init__(self):
                super().__init__(save_path=str(save_path), do_save=True)

            def log(self, epoch, batch, model, rewards, keep, notkept, loss_val):
                super().log(
                    epoch=epoch,
                    batch=batch,
                    model=model,
                    rewards=rewards,
                    keep=keep,
                    notkept=notkept,
                    loss_val=loss_val,
                )
                on_epoch(self, epoch, rewards, loss_val)
                if cancel_event.is_set():
                    on_log("收到停止请求，正在结束当前 epoch…")
                    raise TrainingCancelled()

        return CallbackLogger


def _prepare_run_config(config: TrainConfig, prepared: D.PreparedData | None = None):
    import torch
    from physo.config import config0, config1, config2

    modules = {"config0": config0, "config1": config1, "config2": config2}
    module = modules.get(config.preset, config0)
    base = getattr(module, config.preset if hasattr(module, config.preset) else "config0")
    cfg = copy.deepcopy(base)

    lc = cfg["learning_config"]
    max_length = max(10, int(config.max_length))
    lc["batch_size"] = int(config.batch_size)
    lc["max_time_step"] = max_length
    lc["n_epochs"] = int(config.epochs)
    lc["risk_factor"] = float(config.risk_factor)
    lc["entropy_weight"] = float(config.entropy_weight)
    use_units = bool(
        prepared is not None
        and prepared.use_units
        and prepared.has_units
        and not prepared.normalize
    )
    lc["observe_units"] = use_units
    lc["get_optimizer"] = lambda model: torch.optim.Adam(model.parameters(), lr=float(config.lr))

    # The LBFGS constant optimizer dominates run time on small GPUs.  Expose a
    # single "steps" knob and derive a bounded inner-iteration count from it.
    steps = max(1, int(config.const_opti_steps))
    try:
        method_args = cfg["free_const_opti_args"]["method_args"]
        method_args["n_steps"] = steps
        method_args.setdefault("lbfgs_func_args", {})["max_iter"] = max(1, min(4, steps // 2))
    except Exception:
        pass

    priors = []
    for item in cfg.get("priors_config", []):
        name, args = item if isinstance(item, (tuple, list)) else (item, None)
        args = dict(args or {})
        if name == "HardLengthPrior":
            args["max_length"] = max_length
            args["min_length"] = min(int(args.get("min_length", 4)), max_length)
        elif name == "SoftLengthPrior":
            args["length_loc"] = min(max_length - 2, int(args.get("length_loc", 8)))
            args["scale"] = max(1.0, float(args.get("scale", 5)))
        priors.append((name, args or None))
    cfg["priors_config"] = priors
    cfg["reward_config"]["parallel_mode"] = bool(config.parallel)
    cfg["reward_config"]["n_cpus"] = int(config.n_cpus) if config.parallel else 1
    return cfg


def _rounded_expr(expr, digits: int = 4):
    """Round numeric constants for readable formulas (display only)."""
    import sympy

    try:
        return expr.replace(
            lambda node: isinstance(node, sympy.Float),
            lambda node: sympy.Float(round(float(node), digits)),
        )
    except Exception:
        return expr


def _physical_expression(expr, x_names, x_mean, x_std, y_mean, y_std):
    """Map an expression in normalized space back to the original variables."""
    import sympy

    by_name = {symbol.name: symbol for symbol in expr.free_symbols}
    subs = {}
    for name, mean, std in zip(x_names, x_mean, x_std):
        std = float(std)
        if not math.isfinite(std) or abs(std) < 1e-15:
            continue
        symbol = by_name.get(name) or sympy.Symbol(name, real=True)
        subs[symbol] = (symbol - float(mean)) / std
    return sympy.simplify(expr.subs(subs) * float(y_std) + float(y_mean))


def _make_candidate(program, data: D.PreparedData, reward: float, source: str) -> Candidate:
    import sympy

    # Programs exported from a CUDA run keep their free-constant tensors on the
    # GPU; SymPy needs host scalars when it substitutes them into the expression.
    try:
        program.free_consts.cpu()
    except Exception:
        pass

    try:
        raw = program.get_infix_sympy(evaluate_consts=True)
        if isinstance(raw, (list, tuple, np.ndarray)):
            expr = raw[0]
        else:
            expr = raw
    except Exception:
        # Fall back to the infix string with free constants still symbolic.
        expr = D.expression_to_sympy(program.get_infix_str(), data.x_names)

    infix = str(expr)
    from .mathrender import humanize_expression

    display_source = humanize_expression(_rounded_expr(expr, 4))
    display_expr = str(display_source)
    pretty = safe_pretty(display_source)
    latex = safe_latex(display_source)

    if data.normalize and len(infix) < 700:
        try:
            physical = _physical_expression(
                expr, data.x_names, data.x_mean, data.x_std, data.y_mean, data.y_std
            )
            physical = humanize_expression(_rounded_expr(physical, 4))
            physical_str = str(physical)
            if len(physical_str) <= 1200:
                display_expr = physical_str
                pretty = safe_pretty(physical)
                latex = safe_latex(physical)
        except Exception:
            pass

    pred_train = D.evaluate_expression_on_data(infix, data, data.train_idx)
    m_train = D.regression_metrics(data.y_raw[data.train_idx], pred_train)
    if data.eval_idx.size:
        pred_eval = D.evaluate_expression_on_data(infix, data, data.eval_idx)
        m_eval = D.regression_metrics(data.y_raw[data.eval_idx], pred_eval)
    else:
        m_eval = {"r2": float("nan"), "rmse": float("nan")}

    free_consts: dict[str, float] = {}
    try:
        names = [str(n) for n in program.library.class_free_constants_names]
        values = program.free_consts.class_values.detach().cpu().numpy().ravel()
        free_consts = {name: float(val) for name, val in zip(names, values)}
    except Exception:
        pass

    try:
        complexity = int(round(sum(float(getattr(tok, "complexity", 1.0)) for tok in program.tokens)))
    except Exception:
        complexity = 0
    try:
        length = int(len(program.tokens))
    except Exception:
        length = 0

    return Candidate(
        expression=infix,
        display_expression=display_expr,
        pretty=pretty,
        latex=latex,
        complexity=complexity,
        length=length,
        reward=float(reward) if reward is not None else float("nan"),
        r2_train=float(m_train["r2"]),
        rmse_train=float(m_train["rmse"]),
        r2_eval=float(m_eval["r2"]),
        rmse_eval=float(m_eval["rmse"]),
        source=source,
        free_consts=free_consts,
    )


def safe_pretty(expr) -> str:
    import sympy

    try:
        return sympy.pretty(expr, use_unicode=True)
    except Exception:
        return str(expr)


def safe_latex(expr) -> str:
    import sympy

    try:
        return sympy.latex(expr)
    except Exception:
        return ""


def _reward_from_r2(r2: float) -> float:
    try:
        if not np.isfinite(r2):
            return 0.0
        nr = math.sqrt(max(0.0, 1.0 - float(r2)))
        return 1.0 / (1.0 + nr)
    except Exception:
        return 0.0


def _build_result(
    data: D.PreparedData,
    config: TrainConfig,
    logger,
    best_program,
    run_dir: Path,
    duration: float,
    stopped: bool,
) -> RunResult:
    best_reward = float("nan")
    try:
        best_reward = float(np.asarray(logger.overall_max_R_history[-1]))
    except Exception:
        try:
            best_reward = float(np.asarray(logger.R_history)[-1].max())
        except Exception:
            pass

    candidates: list[Candidate] = []
    by_expression: dict[str, Candidate] = {}

    def add_candidate(program, reward: float | None, source: str) -> None:
        try:
            cand = _make_candidate(program, data, float("nan") if reward is None else reward, source)
        except Exception:
            return
        if reward is None or not np.isfinite(reward):
            cand.reward = _reward_from_r2(cand.r2_train)
        old = by_expression.get(cand.expression)
        if old is not None:
            if cand.r2_train > old.r2_train:
                by_expression[cand.expression] = cand
            return
        by_expression[cand.expression] = cand
        candidates.append(cand)

    # 1) Programs that actually improved the overall best over the run.
    for prog in list(getattr(logger, "hall_of_fame", []) or [])[-12:]:
        add_candidate(prog, None, "history")

    # 2) Best programs of the last epoch (gives a useful complexity/reward trade-off).
    try:
        rewards_last = logger.R
        if hasattr(rewards_last, "detach"):
            rewards_last = rewards_last.detach().cpu().numpy()
        rewards_last = np.asarray(rewards_last, dtype=float).ravel()
        programs_last = list(logger.programs_epoch)
        order = np.argsort(-rewards_last)[:20]
        for idx in order:
            if 0 <= int(idx) < len(programs_last):
                add_candidate(programs_last[int(idx)], float(rewards_last[int(idx)]), "pareto")
    except Exception:
        pass

    # 3) PhySO's own Pareto front (may be unavailable depending on version).
    try:
        _complexities, programs, rewards, _rmse = logger.get_pareto_front()
        for prog, reward in list(zip(programs, rewards))[-20:]:
            add_candidate(prog, float(reward), "pareto")
    except Exception:
        pass

    # 4) Whatever physo.SR returned as its best expression.
    if best_program is not None:
        add_candidate(best_program, best_reward, "best")

    if not candidates:
        raise RuntimeError("训练结束但没有得到可用的表达式（可能所有候选都因数值问题被丢弃）。")

    candidates = [c for c in by_expression.values() if np.isfinite(c.r2_train)]
    if not candidates:
        raise RuntimeError("训练结束但没有得到数值有效的表达式。")

    # Highest training R² is the best result, independent of what the library labels as best.
    best_index = max(range(len(candidates)), key=lambda i: (candidates[i].r2_train, -candidates[i].complexity))
    for cand in candidates:
        cand.source = "pareto"
    candidates[best_index].source = "best"
    candidates[best_index].reward = best_reward if np.isfinite(best_reward) else _reward_from_r2(candidates[best_index].r2_train)

    candidates.sort(key=lambda c: (c.complexity, -c.r2_train))
    best_expr = next(c.expression for c in candidates if c.source == "best")
    candidates = candidates[:30]
    best_index = next((i for i, c in enumerate(candidates) if c.expression == best_expr), 0)
    # Make sure the best is inside the 30-item window.
    if all(c.source != "best" for c in candidates):
        best_index = max(range(len(candidates)), key=lambda i: candidates[i].r2_train)
        candidates[best_index].source = "best"

    X_train = data.X_raw[:, data.train_idx]
    y_train = data.y_raw[data.train_idx]
    if data.eval_idx.size:
        X_eval = data.X_raw[:, data.eval_idx]
        y_eval = data.y_raw[data.eval_idx]
    else:
        X_eval = np.empty((data.n_dim, 0))
        y_eval = np.empty((0,))

    result = RunResult(
        summary=data.summary_dict(),
        config=config.to_dict(),
        candidates=candidates,
        active_index=best_index,
        history=list(getattr(logger, "_gui_history", [])),
        run_dir=str(run_dir),
        created_at=datetime.now().isoformat(timespec="seconds"),
        duration=duration,
        epochs_done=int(getattr(logger, "epoch", 0)) + 1 if getattr(logger, "epoch", None) is not None else 0,
        stopped=stopped,
        data=data,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
    )
    return result


# --------------------------------------------------------------------------- #
# Training worker
# --------------------------------------------------------------------------- #
class TrainingWorker(QThread):
    progress = Signal(dict)
    log_line = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, prepared: D.PreparedData, config: TrainConfig, run_dir: Path, parent=None):
        super().__init__(parent)
        self.prepared = prepared
        self.config = config
        self.run_dir = Path(run_dir)
        self.cancel_event = threading.Event()
        self._log_path = self.run_dir / "train.log"
        self._logger = None

    # -- helpers ------------------------------------------------------------ #
    def request_stop(self) -> None:
        self.cancel_event.set()

    def _write_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        self.log_line.emit(line)

    def _on_epoch(self, logger, epoch: int, rewards, loss_val) -> None:
        try:
            best_reward = float(np.asarray(logger.overall_max_R_history[-1]))
        except Exception:
            best_reward = float("nan")
        try:
            max_reward = float(np.asarray(rewards.max()))
            mean_reward = float(np.asarray(rewards.mean()))
        except Exception:
            max_reward = mean_reward = float("nan")
        try:
            loss = float(np.asarray(loss_val))
        except Exception:
            loss = float("nan")
        best_r2 = D.reward_to_r2(best_reward)
        max_r2 = D.reward_to_r2(max_reward)
        mean_r2 = D.reward_to_r2(mean_reward)
        expression = ""
        try:
            expression = logger.overall_best_prog_str_history[-1]
        except Exception:
            expression = ""
        if expression and expression != getattr(self, "_latex_expr", None):
            try:
                from .mathrender import latex_from_infix

                self._latex_value = latex_from_infix(expression, self.prepared.x_names)
                self._latex_expr = expression
            except Exception:
                self._latex_value = ""
        best_latex = getattr(self, "_latex_value", "")
        try:
            best_complexity = int(np.asarray(logger.best_prog_complexity_history[-1]))
        except Exception:
            best_complexity = 0
        try:
            best_length = int(getattr(logger.best_prog, "size", 0))
        except Exception:
            best_length = 0
        try:
            mean_complexity = float(np.asarray(logger.mean_complexity_history[-1]))
        except Exception:
            mean_complexity = float("nan")
        try:
            n_physical = int(np.asarray(logger.n_physical[-1]))
            n_rewarded = int(np.asarray(logger.n_rewarded[-1]))
        except Exception:
            n_physical = n_rewarded = 0
        now = time.time()
        elapsed = now - getattr(self, "_t0", now)
        epoch_time = now - getattr(self, "_last_epoch_ts", getattr(self, "_t0", now))
        self._last_epoch_ts = now
        remaining = max(0, int(self.config.epochs) - int(epoch) - 1)
        eta = epoch_time * remaining
        history = getattr(logger, "_gui_history", None)
        if history is None:
            history = []
            logger._gui_history = history
        entry = {
            "epoch": int(epoch),
            "best_reward": best_reward,
            "best_r2": best_r2,
            "max_reward": max_reward,
            "max_r2": max_r2,
            "mean_reward": mean_reward,
            "mean_r2": mean_r2,
            "loss": loss,
            "best_complexity": best_complexity,
            "best_length": best_length,
            "mean_complexity": mean_complexity,
            "n_physical": n_physical,
            "n_rewarded": n_rewarded,
            "epoch_time": epoch_time,
            "elapsed": elapsed,
            "eta": eta,
            "expression": expression,
            "best_latex": best_latex,
        }
        history.append(entry)
        payload = {
            "epoch": int(epoch),
            "epochs": int(self.config.epochs),
            "best_reward": best_reward,
            "best_r2": best_r2,
            "max_r2": max_r2,
            "mean_r2": mean_r2,
            "best_complexity": best_complexity,
            "best_length": best_length,
            "mean_complexity": mean_complexity,
            "n_physical": n_physical,
            "n_rewarded": n_rewarded,
            "loss": loss,
            "expression": expression,
            "best_latex": best_latex,
            "history": history,
            "elapsed": elapsed,
            "epoch_time": epoch_time,
            "eta": eta,
        }
        self.progress.emit(payload)
        self._write_log(
            f"epoch {epoch + 1:>3}/{self.config.epochs} · best R={best_reward:.5f} · "
            f"best R²={D.format_number(best_r2, 4)} · len={len(expression)}"
        )

    # -- main --------------------------------------------------------------- #
    def run(self) -> None:  # noqa: C901 - orchestration is inherently linear
        self._t0 = time.time()
        self._write_log("启动训练任务…")
        self._write_log(
            f"数据：{self.prepared.name} · {self.prepared.describe()} · 目标 {self.prepared.y_name}"
        )
        try:
            import torch
            import physo
            import physo.learn.monitoring as monitoring
            from physo.physym import functions as physo_functions
        except Exception as exc:
            self.failed.emit(
                "无法导入 PhySO / PyTorch。请在 conda 环境中运行 `pip install -r requirements.txt`。\n"
                f"详细信息：{D.friendly_error(exc)}"
            )
            return

        try:
            seed = int(self.config.seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if int(self.config.cpu_threads) > 0:
                try:
                    torch.set_num_threads(int(self.config.cpu_threads))
                except Exception:
                    pass
            try:
                self._write_log(f"CPU 线程：{torch.get_num_threads()}")
            except Exception:
                pass

            device = self.config.resolve_device()
            if device == "cuda" and not torch.cuda.is_available():
                self._write_log("未检测到可用 CUDA，自动切换到 CPU。")
                device = "cpu"
            self._write_log(f"计算设备：{device.upper()}")
            if device == "cuda":
                try:
                    self._write_log(
                        f"GPU：{torch.cuda.get_device_name(0)} · "
                        f"显存 {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
                    )
                except Exception:
                    pass

            cfg = _prepare_run_config(self.config, self.prepared)
            X_train, y_train, X_eval, y_eval = self.prepared.model_arrays()
            del X_eval, y_eval
            self._write_log(
                f"训练样本 {X_train.shape[1]} 个 · 输入维度 {X_train.shape[0]} · "
                f"batch {self.config.batch_size} · epochs {self.config.epochs}"
            )
            self._write_log(
                "量纲分析：" + ("开启" if cfg["learning_config"].get("observe_units") else "关闭")
            )

            ops = [op for op in self.config.ops if op in physo_functions.OPS_NAMES]
            if not ops:
                raise ValueError("至少选择一个有效算子。")

            fixed = [float(v) for v in self.config.fixed_consts if np.isfinite(float(v))]
            free_names = [f"c{i}" for i in range(max(0, int(self.config.n_free_consts)))]
            zero_units = [[0.0] * 7 for _ in range(self.prepared.n_dim)]
            x_units = self.prepared.x_units if self.prepared.use_units else zero_units
            y_units = self.prepared.y_units if self.prepared.use_units else [0.0] * 7

            cancel = self.cancel_event
            on_log = self._write_log
            on_epoch = self._on_epoch

            def make_logger():
                logger = _CallbackLogger.build(
                    monitoring,
                    self.run_dir / "SR.log",
                    on_epoch,
                    cancel,
                    on_log,
                )()
                self._logger = logger
                return logger

            best_expression = None
            stopped = False
            try:
                best_expression, _run_logger = physo.SR(
                    X_train,
                    y_train,
                    X_names=self.prepared.x_names,
                    X_units=x_units,
                    y_name=self.prepared.y_name,
                    y_units=y_units,
                    fixed_consts=fixed,
                    fixed_consts_units=[[0.0] * 7 for _ in fixed],
                    free_consts_names=free_names if free_names else [],
                    free_consts_units=[[0.0] * 7 for _ in free_names],
                    op_names=ops,
                    use_protected_ops=True,
                    stop_reward=float(self.config.stop_reward),
                    stop_after_n_epochs=int(self.config.stop_after_n_epochs),
                    max_n_evaluations=self.config.max_n_evaluations,
                    epochs=int(self.config.epochs),
                    run_config=cfg,
                    get_run_logger=make_logger,
                    get_run_visualiser=lambda: _NullVisualiser(),
                    parallel_mode=bool(self.config.parallel),
                    n_cpus=int(self.config.n_cpus) if self.config.parallel else None,
                    device=device,
                )
            except TrainingCancelled:
                stopped = True
                self._write_log("训练已停止，正在整理当前最优结果…")
            except Exception as exc:
                self._write_log("训练过程中出现错误：")
                self._write_log(traceback.format_exc())
                self.failed.emit(
                    "训练失败："
                    + D.friendly_error(exc)
                    + "\n\n提示：若显存不足，请降低 batch size / 表达式长度，或在模型页选择 CPU。"
                )
                return

            logger = self._logger
            if logger is None:
                raise RuntimeError("训练没有产生日志对象。")
            if best_expression is None:
                try:
                    best_expression = logger.best_prog
                except Exception:
                    best_expression = None

            duration = time.time() - self._t0
            try:
                result = _build_result(
                    self.prepared,
                    self.config,
                    logger,
                    best_expression,
                    self.run_dir,
                    duration,
                    stopped,
                )
            except Exception as exc:
                self._write_log(traceback.format_exc())
                self.failed.emit("无法整理训练结果：" + D.friendly_error(exc))
                return

            self._write_log(
                f"训练完成 · {result.epochs_done} epochs · 用时 {duration:.1f}s · "
                f"最佳 R²={D.format_number(result.active.r2_train, 5)}"
            )
            try:
                result.save_to_dir(self.run_dir)
                self._write_log(f"结果已保存：{self.run_dir}")
            except Exception as exc:
                self._write_log("结果自动保存失败：" + D.friendly_error(exc))
            # Copy our training log into the run folder next to PhySO's SR.log.
            self.succeeded.emit(result)
        except Exception as exc:
            self._write_log(traceback.format_exc())
            self.failed.emit("训练初始化失败：" + D.friendly_error(exc))
