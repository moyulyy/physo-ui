"""LaTeX formula rendering for the Qt UI (no external browser/KaTeX needed)."""

from __future__ import annotations

import io
import re
from functools import lru_cache

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["mathtext.default"] = "it"

# Matplotlib's mathtext understands less LaTeX than SymPy emits.  These small
# rewrites cover the constructs that appear in PhySO expressions.
_OPERATOR_RE = re.compile(r"\\operatorname\{([^{}]*)\}")
_TEXT_RE = re.compile(r"\\text\{([^{}]*)\}")


def clean_latex(latex: str) -> str:
    text = (latex or "").strip()
    text = text.replace("$", "")
    text = text.replace("\\mathdefault", "")
    text = text.replace("\\displaystyle", "")
    text = _OPERATOR_RE.sub(lambda m: r"\mathrm{" + m.group(1) + "}", text)
    text = _TEXT_RE.sub(lambda m: r"\mathrm{" + m.group(1) + "}", text)
    # SymPy sometimes emits \left. / \right. without a delimiter; mathtext dislikes them.
    text = text.replace("\\left.", "").replace("\\right.", "")
    return text


@lru_cache(maxsize=256)
def _png_bytes(latex: str, color: str, fontsize: float, dpi: int = 200) -> bytes | None:
    cleaned = clean_latex(latex)
    if not cleaned:
        return None
    try:
        fig = Figure(figsize=(0.4, 0.2), dpi=dpi, facecolor="none")
        FigureCanvasAgg(fig)
        fig.text(0.01, 0.02, f"${cleaned}$", color=color, fontsize=fontsize)
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=dpi,
        )
        data = buffer.getvalue()
        return data if len(data) > 100 else None
    except Exception:
        return None


def latex_pixmap(latex: str, color: str = theme.INK, fontsize: float = 18) -> QPixmap | None:
    data = _png_bytes(latex, color, float(fontsize))
    if not data:
        return None
    pix = QPixmap()
    if not pix.loadFromData(data, "PNG"):
        return None
    pix.setDevicePixelRatio(2.0)
    return pix


class MathLabel(QWidget):
    """A width-aware widget painting a LaTeX formula, with a text fallback."""

    def __init__(
        self,
        latex: str = "",
        *,
        color: str = theme.INK,
        fontsize: int = 18,
        min_height: int | None = None,
        fallback: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._latex = ""
        self._color = color
        self._fontsize = fontsize
        self._fallback = fallback
        self._pixmap: QPixmap | None = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(min_height if min_height is not None else int(fontsize * 2.6))
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.set_latex(latex, color=color, fontsize=fontsize, fallback=fallback)

    def set_latex(
        self,
        latex: str,
        *,
        color: str | None = None,
        fontsize: int | None = None,
        fallback: str | None = None,
    ) -> None:
        self._latex = latex or ""
        if color is not None:
            self._color = color
        if fontsize is not None:
            self._fontsize = fontsize
        if fallback is not None:
            self._fallback = fallback
        self._pixmap = latex_pixmap(self._latex, self._color, self._fontsize) if self._latex else None
        self.setToolTip(self._latex or self._fallback)
        self.updateGeometry()
        self.update()

    def clear_formula(self, text: str = "") -> None:
        self._latex = ""
        self._pixmap = None
        self._fallback = text
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        if self._pixmap is not None:
            ratio = self._pixmap.height() / max(1, self._pixmap.width())
            height = int(min(120, max(self.minimumHeight(), 520 * ratio)))
            return QSize(520, height)
        return QSize(520, max(self.minimumHeight(), 30))

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self._pixmap is not None and not self._pixmap.isNull():
            pw = max(1.0, self._pixmap.width() / self._pixmap.devicePixelRatio())
            ph = max(1.0, self._pixmap.height() / self._pixmap.devicePixelRatio())
            available_w = max(1, self.width())
            available_h = max(1, self.height())
            scale = min(available_w / pw, available_h / ph, 1.35)
            target_w = pw * scale
            target_h = ph * scale
            x = (self.width() - target_w) / 2
            y = (self.height() - target_h) / 2
            p.drawPixmap(
                int(x),
                int(y),
                int(target_w),
                int(target_h),
                self._pixmap,
            )
        else:
            text = self._fallback or self._latex
            if text:
                font = QFont(theme.mono_font(10))
                font.setPointSize(max(9, min(13, self._fontsize - 4)))
                p.setFont(font)
                p.setPen(QColor(self._color))
                p.drawText(self.rect().adjusted(4, 2, -4, -2), Qt.AlignCenter | Qt.TextWordWrap, text)
        p.end()


def humanize_expression(expr):
    """Turn PhySO-specific operators into standard mathematical notation.

    The numeric expression used for inference is untouched; this form is only
    for display, so ``n2(x)`` becomes ``x²``, ``logabs(x)`` becomes ``log|x|``,
    and so on.  Standard ``frac``/``sqrt``/``exp``/``log``/trig functions are
    already emitted by SymPy's LaTeX printer.
    """
    import sympy

    funcs = {
        "n2": lambda x: x**2,
        "n3": lambda x: x**3,
        "n4": lambda x: x**4,
        "inv": lambda x: 1 / x,
        "cbrt": lambda x: sympy.Pow(x, sympy.Rational(1, 3)),
        "logabs": lambda x: sympy.log(sympy.Abs(x)),
        "expneg": lambda x: sympy.exp(-x),
        "harmonic": lambda x: 1 / x,
        "sigmoid": lambda x: 1 / (1 + sympy.exp(-x)),
        "max": lambda x, y: sympy.Max(x, y),
        "min": lambda x, y: sympy.Min(x, y),
    }

    def is_custom(node):
        return isinstance(node, sympy.Function) and node.func.__name__ in funcs

    def replace(node):
        fn = funcs[node.func.__name__]
        try:
            return fn(*node.args)
        except Exception:
            return node

    try:
        return expr.replace(is_custom, replace)
    except Exception:
        return expr


def latex_from_infix(expression: str, x_names: list[str]) -> str:
    """Best-effort infix -> LaTeX conversion (free constants such as c0 stay symbolic)."""
    try:
        import sympy

        local = {name: sympy.Symbol(name, real=True) for name in x_names}
        expr = sympy.parse_expr(expression, local_dict=local, evaluate=False)
        expr = humanize_expression(expr)
        return sympy.latex(expr)
    except Exception:
        return ""
