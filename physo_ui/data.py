"""Data loading, validation, preparation and symbolic-expression evaluation."""

from __future__ import annotations

import io
import json
import keyword
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"

#: Symbols that would collide with SymPy / Python names when used as variables.
RESERVED_NAMES = {
    "E", "I", "N", "S", "O", "Q", "pi", "oo", "nan", "zoo", "lambda", "beta", "gamma",
    "zeta", "eta", "theta", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi",
    "omega", "exp", "log", "sin", "cos", "tan", "sqrt", "cbrt", "inv", "pow", "abs",
    "max", "min", "erf", "tanh", "sinh", "cosh", "arcsin", "arccos", "arctan", "n2",
    "n3", "n4", "logabs", "expneg", "sigmoid", "harmonic", "true", "false", "none",
}

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".dat", ".npy", ".npz", ".xlsx", ".xls"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class RawDataset:
    name: str
    frame: pd.DataFrame
    path: str | None = None
    source: str = "file"

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_cols(self) -> int:
        return len(self.frame.columns)


def _looks_numeric(value: Any) -> bool:
    try:
        float(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False


def _read_delimited(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("文件内容为空。")
    sample = "\n".join(lines[:5])
    first = lines[0]
    candidates = []
    for sep in [None, ",", "\t", ";", "|", " "]:
        try:
            probe = pd.read_csv(
                io.StringIO(sample),
                sep=sep,
                engine="python",
                header=None,
                nrows=5,
                comment="#",
            )
            if probe.shape[1] >= 2:
                candidates.append((probe.shape[1], sep, probe))
        except Exception:
            continue
    if not candidates:
        raise ValueError("无法识别文本文件的分隔符。")
    best_cols, sep, probe = max(candidates, key=lambda item: item[0])
    del best_cols
    has_header = not all(_looks_numeric(v) for v in probe.iloc[0].tolist())
    header = 0 if has_header else None
    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",
        header=header,
        comment="#",
        skip_blank_lines=True,
    )
    if header is None:
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_array(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 1:
            return pd.DataFrame({"x0": np.arange(arr.shape[0]), "y": arr})
        if arr.ndim == 2:
            cols = [f"x{i}" for i in range(arr.shape[1] - 1)] + ["y"]
            return pd.DataFrame(arr, columns=cols[: arr.shape[1]])
        raise ValueError("只支持 1D 或 2D 的 .npy 数组。")
    npz = np.load(path, allow_pickle=False)
    keys = list(npz.keys())
    lower = {k.lower(): k for k in keys}
    if "x" in lower and ("y" in lower or "target" in lower):
        X = npz[lower["x"]]
        y = npz[lower.get("y", lower.get("target"))]
        X = np.atleast_2d(X)
        if X.shape[0] != len(y) and X.shape[1] == len(y):
            X = X.T
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        df["y"] = np.asarray(y).ravel()
        return df
    arrays = [npz[k] for k in keys]
    arr = max(arrays, key=lambda a: np.asarray(a).size)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return pd.DataFrame({"x0": np.arange(arr.shape[0]), "y": arr})
    if arr.ndim != 2:
        raise ValueError("无法从 .npz 文件中推断数据表结构。")
    cols = [f"x{i}" for i in range(arr.shape[1] - 1)] + ["y"]
    return pd.DataFrame(arr, columns=cols)


def load_table(path: str | Path) -> RawDataset:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix}")
    if suffix in {".csv", ".tsv", ".txt", ".dat"}:
        df = _read_delimited(path)
    elif suffix in {".npy", ".npz"}:
        df = _read_array(path)
    else:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("数据表为空。")
    # Drop fully empty unnamed columns created by trailing separators.
    keep = [c for c in df.columns if not (str(c).startswith("Unnamed:") and df[c].isna().all())]
    df = df[keep]
    return RawDataset(name=path.stem, frame=df, path=str(path), source="file")


def load_bundled(filename: str) -> RawDataset:
    path = DATASETS_DIR / filename
    raw = load_table(path)
    meta = dataset_library().get(filename, {})
    raw.name = meta.get("name", path.stem)
    raw.source = "bundled"
    return raw


def dataset_library() -> dict[str, dict[str, Any]]:
    index_path = DATASETS_DIR / "index.json"
    if not index_path.exists():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = data.get("datasets", data if isinstance(data, list) else [])
    return {item.get("file"): item for item in items if item.get("file")}


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
def sanitize_names(columns: list[str]) -> dict[str, str]:
    """Map arbitrary column labels to unique, SymPy-friendly variable names."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for i, col in enumerate(columns):
        base = re.sub(r"[^0-9A-Za-z_]+", "_", str(col), flags=re.ASCII).strip("_")
        if not base:
            base = f"x{i}"
        if base[0].isdigit():
            base = f"x_{base}"
        if base in RESERVED_NAMES or keyword.iskeyword(base) or base.startswith("_"):
            base = f"{base}_v"
        name = base
        counter = 2
        while name in used or name in RESERVED_NAMES:
            name = f"{base}_{counter}"
            counter += 1
        used.add(name)
        mapping[col] = name
    return mapping


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    name: str
    frame: pd.DataFrame
    x_cols: list[str]
    y_col: str
    x_names: list[str]
    y_name: str
    X_raw: np.ndarray
    y_raw: np.ndarray
    split_enabled: bool = True
    train_fraction: float = 0.8
    normalize: bool = False
    x_units: list[list[float]] = field(default_factory=list)
    y_units: list[float] = field(default_factory=list)
    use_units: bool = True
    train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    eval_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    x_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    x_std: np.ndarray = field(default_factory=lambda: np.array([]))
    y_mean: float = 0.0
    y_std: float = 1.0
    source_path: str | None = None
    dropped_rows: int = 0

    def __post_init__(self) -> None:
        if self.X_raw.ndim == 1:
            self.X_raw = self.X_raw.reshape(1, -1)
        if not self.train_idx.size:
            order = np.arange(self.y_raw.shape[0])
            if self.split_enabled and len(order) > 4:
                rng = np.random.default_rng(42)
                rng.shuffle(order)
                n_train = max(2, int(round(len(order) * np.clip(self.train_fraction, 0.3, 1.0))))
                n_train = min(n_train, len(order) - 1)
                self.train_idx = np.sort(order[:n_train])
                self.eval_idx = np.sort(order[n_train:])
            else:
                self.train_idx = order
                self.eval_idx = np.array([], dtype=int)
        if len(self.x_units) != self.X_raw.shape[0]:
            self.x_units = [[0.0] * 7 for _ in range(self.X_raw.shape[0])]
        if len(self.y_units) != 7:
            self.y_units = [0.0] * 7
        if not self.x_mean.size:
            self.x_mean = np.zeros(self.X_raw.shape[0], dtype=float)
            self.x_std = np.ones(self.X_raw.shape[0], dtype=float)
            if self.normalize:
                self.x_mean = np.nanmean(self.X_raw[:, self.train_idx], axis=1)
                std = np.nanstd(self.X_raw[:, self.train_idx], axis=1)
                self.x_std = np.where(std > 0, std, 1.0)
                self.y_mean = float(np.nanmean(self.y_raw[self.train_idx]))
                y_std = float(np.nanstd(self.y_raw[self.train_idx]))
                self.y_std = y_std if y_std > 0 else 1.0

    # ---- shape helpers ---------------------------------------------------- #
    @property
    def n_samples(self) -> int:
        return int(self.y_raw.shape[0])

    @property
    def n_dim(self) -> int:
        return int(self.X_raw.shape[0])

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_eval(self) -> int:
        return int(self.eval_idx.size)

    @property
    def name_map(self) -> dict[str, str]:
        return {orig: sanitized for orig, sanitized in zip(self.x_cols, self.x_names)}

    @property
    def has_units(self) -> bool:
        return any(abs(float(v)) > 1e-12 for vector in self.x_units for v in vector) or any(
            abs(float(v)) > 1e-12 for v in self.y_units
        )

    # ---- scaling ---------------------------------------------------------- #
    def transform_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1) if X.size == self.n_dim else X.reshape(1, -1)
        elif X.ndim == 2 and X.shape[0] != self.n_dim and X.shape[1] == self.n_dim:
            X = X.T
        if not self.normalize:
            return X
        return (X - self.x_mean.reshape(-1, 1)) / self.x_std.reshape(-1, 1)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if not self.normalize:
            return y
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if not self.normalize:
            return y
        return y * self.y_std + self.y_mean

    def model_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train = self.transform_X(self.X_raw[:, self.train_idx])
        y_train = self.transform_y(self.y_raw[self.train_idx])
        if self.eval_idx.size:
            X_eval = self.transform_X(self.X_raw[:, self.eval_idx])
            y_eval = self.transform_y(self.y_raw[self.eval_idx])
        else:
            X_eval = np.empty((self.n_dim, 0))
            y_eval = np.empty((0,))
        return X_train, y_train, X_eval, y_eval

    def describe(self) -> str:
        split = f"训练 {self.n_train} · 评估 {self.n_eval}" if self.n_eval else f"全部 {self.n_train} 行"
        norm = " · 已标准化" if self.normalize else ""
        units = " · 量纲分析" if (self.use_units and self.has_units) else ""
        return f"{self.n_dim} 个输入变量 · {split}{norm}{units}"

    def summary_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "x_columns": self.x_cols,
            "y_column": self.y_col,
            "x_names": self.x_names,
            "y_name": self.y_name,
            "n_samples": self.n_samples,
            "n_dim": self.n_dim,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "split_enabled": self.split_enabled,
            "train_fraction": self.train_fraction,
            "normalize": self.normalize,
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "x_median": np.nanmedian(self.X_raw, axis=1).tolist(),
            "x_min": np.nanmin(self.X_raw, axis=1).tolist(),
            "x_max": np.nanmax(self.X_raw, axis=1).tolist(),
            "x_units": self.x_units,
            "y_units": self.y_units,
            "use_units": self.use_units,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "dropped_rows": self.dropped_rows,
            "y_min": float(np.nanmin(self.y_raw)),
            "y_max": float(np.nanmax(self.y_raw)),
        }


def prepare_data(
    raw: RawDataset,
    x_cols: list[str],
    y_col: str,
    *,
    train_fraction: float = 0.8,
    split_enabled: bool = True,
    normalize: bool = False,
    x_units: list[list[float]] | None = None,
    y_units: list[float] | None = None,
    use_units: bool = True,
) -> PreparedData:
    if not x_cols:
        raise ValueError("请至少选择一个输入变量 X。")
    if y_col in x_cols:
        raise ValueError("目标变量 y 不能同时作为输入变量。")
    missing = [c for c in [*x_cols, y_col] if c not in raw.frame.columns]
    if missing:
        raise ValueError(f"数据中找不到列：{', '.join(missing)}")
    subset = raw.frame[[*x_cols, y_col]].copy()
    for col in subset.columns:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    before = len(subset)
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
    dropped = before - len(subset)
    if len(subset) < 8:
        raise ValueError("有效数据不足 8 行，请检查缺失值或列类型。")
    X = subset[x_cols].to_numpy(dtype=float).T
    y = subset[y_col].to_numpy(dtype=float).ravel()
    if np.allclose(np.std(y), 0.0):
        raise ValueError("目标变量 y 为常数，无法进行符号回归。")
    mapping = sanitize_names([*x_cols, y_col])
    x_names = [mapping[c] for c in x_cols]
    y_name = mapping[y_col]
    return PreparedData(
        name=raw.name,
        frame=subset,
        x_cols=list(x_cols),
        y_col=y_col,
        x_names=x_names,
        y_name=y_name,
        X_raw=X,
        y_raw=y,
        split_enabled=split_enabled,
        train_fraction=train_fraction,
        normalize=normalize,
        x_units=x_units if x_units is not None else [[0.0] * 7 for _ in x_cols],
        y_units=y_units if y_units is not None else [0.0] * 7,
        use_units=use_units,
        source_path=raw.path,
        dropped_rows=dropped,
    )


# --------------------------------------------------------------------------- #
# Expression evaluation
# --------------------------------------------------------------------------- #
def _custom_modules() -> dict[str, Any]:
    from scipy import special  # scipy is a scikit-learn dependency

    def _safe_vec(fn):
        return np.vectorize(fn, otypes=[float])

    return {
        "n2": np.square,
        "n3": lambda x: np.power(x, 3),
        "n4": lambda x: np.power(x, 4),
        "cbrt": lambda x: np.cbrt(x),
        "inv": lambda x: 1.0 / x,
        "logabs": lambda x: np.log(np.abs(x)),
        "expneg": lambda x: np.exp(-x),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "harmonic": lambda x: 1.0 / x,
        "erf": _safe_vec(math.erf),
        "max": np.maximum,
        "min": np.minimum,
        "abs": np.abs,
        "arctan": np.arctan,
        "arccos": np.arccos,
        "arcsin": np.arcsin,
        "tanh": np.tanh,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "special_erf": _safe_vec(special.erf),
    }


def expression_to_sympy(expression: str, x_names: list[str]):
    import sympy

    local = {name: sympy.Symbol(name, real=True) for name in x_names}
    expr = sympy.parse_expr(expression, local_dict=local, evaluate=False)
    return expr


def expression_to_callable(expression: str, x_names: list[str]):
    import sympy

    expr = expression_to_sympy(expression, x_names)
    symbols = [sympy.Symbol(name, real=True) for name in x_names]
    func = sympy.lambdify(symbols, expr, modules=[_custom_modules(), "numpy"])
    return func


def predict_expression(expression: str, x_names: list[str], X_raw: np.ndarray) -> np.ndarray:
    func = expression_to_callable(expression, x_names)
    X = np.asarray(X_raw, dtype=float)
    if X.ndim == 1:
        if len(x_names) == 1:
            X = X.reshape(1, -1)
        elif X.size == len(x_names):
            X = X.reshape(-1, 1)
        else:
            raise ValueError(
                f"输入维度不匹配：需要 {len(x_names)} 个变量，收到长度为 {X.size} 的一维数组。"
            )
    elif X.ndim == 2 and X.shape[0] != len(x_names) and X.shape[1] == len(x_names):
        # Accept (n_samples, n_dim) matrices as well as PhySO's (n_dim, n_samples).
        X = X.T
    args = [X[i] for i in range(len(x_names))]
    with np.errstate(all="ignore"):
        try:
            out = func(*args)
        except Exception as exc:
            raise ValueError(f"表达式求值失败：{exc}") from exc
    out = np.asarray(out)
    if np.iscomplexobj(out):
        out = np.where(np.abs(out.imag) < 1e-10, out.real, np.nan)
    out = np.asarray(out, dtype=float)
    if out.ndim == 0:
        out = np.full(X.shape[1], float(out))
    out = out.ravel()
    if out.size == 1 and X.shape[1] != 1:
        out = np.full(X.shape[1], out.item())
    return out


def evaluate_expression_on_data(
    expression: str,
    data: PreparedData,
    index: np.ndarray | None = None,
) -> np.ndarray:
    if index is None:
        X = data.X_raw
    else:
        X = data.X_raw[:, index]
    X_model = data.transform_X(X)
    pred_model = predict_expression(expression, data.x_names, X_model)
    return data.inverse_y(pred_model)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "max_abs": float("nan"), "n": 0}
    t, p = y_true[mask], y_pred[mask]
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "max_abs": float(np.max(np.abs(t - p))),
        "n": int(mask.sum()),
    }


def reward_to_r2(reward: float) -> float:
    try:
        r = float(reward)
        if r <= 0 or not np.isfinite(r):
            return float("nan")
        return 2.0 / r - (1.0 / r) ** 2
    except Exception:
        return float("nan")


def format_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "—"
    if v != 0 and (abs(v) >= 1e5 or abs(v) < 1e-3):
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}g}"


def friendly_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = text.replace("\n", " ")
    return text[:500]
