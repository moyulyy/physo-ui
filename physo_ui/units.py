"""Physical-dimension helpers and an iOS-styled unit selector."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .widgets import label, section_title


DIMENSIONS = [
    ("L", "长度"),
    ("M", "质量"),
    ("T", "时间"),
    ("I", "电流"),
    ("Θ", "温度"),
    ("N", "物质的量"),
    ("J", "光强"),
]
ZERO_UNIT = [0.0] * len(DIMENSIONS)


@dataclass(frozen=True)
class UnitPreset:
    label: str
    vector: tuple[float, ...]
    symbol: str


def _v(*values: float) -> tuple[float, ...]:
    data = [0.0] * len(DIMENSIONS)
    for i, value in enumerate(values):
        data[i] = float(value)
    return tuple(data)


UNIT_PRESETS: list[UnitPreset] = [
    UnitPreset("无量纲", _v(), "—"),
    UnitPreset("长度 L", _v(1), "L"),
    UnitPreset("面积 L²", _v(2), "L²"),
    UnitPreset("体积 L³", _v(3), "L³"),
    UnitPreset("质量 M", _v(0, 1), "M"),
    UnitPreset("时间 T", _v(0, 0, 1), "T"),
    UnitPreset("电流 I", _v(0, 0, 0, 1), "I"),
    UnitPreset("温度 Θ", _v(0, 0, 0, 0, 1), "Θ"),
    UnitPreset("物质的量 N", _v(0, 0, 0, 0, 0, 1), "N"),
    UnitPreset("光强 J", _v(0, 0, 0, 0, 0, 0, 1), "J"),
    UnitPreset("速度 L·T⁻¹", _v(1, 0, -1), "L·T⁻¹"),
    UnitPreset("加速度 L·T⁻²", _v(1, 0, -2), "L·T⁻²"),
    UnitPreset("频率 T⁻¹", _v(0, 0, -1), "T⁻¹"),
    UnitPreset("动量 M·L·T⁻¹", _v(1, 1, -1), "M·L·T⁻¹"),
    UnitPreset("力 M·L·T⁻²", _v(1, 1, -2), "M·L·T⁻²"),
    UnitPreset("能量 M·L²·T⁻²", _v(2, 1, -2), "M·L²·T⁻²"),
    UnitPreset("功率 M·L²·T⁻³", _v(2, 1, -3), "M·L²·T⁻³"),
    UnitPreset("压强 M·L⁻¹·T⁻²", _v(-1, 1, -2), "M·L⁻¹·T⁻²"),
    UnitPreset("密度 M·L⁻³", _v(-3, 1), "M·L⁻³"),
    UnitPreset("弹簧常量 M·T⁻²", _v(0, 1, -2), "M·T⁻²"),
    UnitPreset("电荷 I·T", _v(0, 0, 1, 1), "I·T"),
    UnitPreset("电压 M·L²·T⁻³·I⁻¹", _v(2, 1, -3, -1), "M·L²·T⁻³·I⁻¹"),
    UnitPreset("电阻 M·L²·T⁻³·I⁻²", _v(2, 1, -3, -2), "M·L²·T⁻³·I⁻²"),
]


_SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _sup(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value))).translate(_SUPERS)
    return f"^{value:g}"


def format_units(vector) -> str:
    if vector is None:
        return "—"
    try:
        values = [float(v) for v in vector]
    except (TypeError, ValueError):
        return "—"
    if len(values) < len(DIMENSIONS):
        values = values + [0.0] * (len(DIMENSIONS) - len(values))
    parts = []
    for (symbol, _name), exponent in zip(DIMENSIONS, values):
        if abs(exponent) < 1e-9:
            continue
        parts.append(symbol if abs(exponent - 1) < 1e-9 else f"{symbol}{_sup(exponent)}")
    return "·".join(parts) if parts else "无量纲"


def unit_vector(value) -> list[float]:
    data = [0.0] * len(DIMENSIONS)
    if value is None:
        return data
    for i, item in enumerate(list(value)[: len(DIMENSIONS)]):
        try:
            data[i] = float(item)
        except (TypeError, ValueError):
            data[i] = 0.0
    return data


def find_preset(vector) -> UnitPreset | None:
    data = unit_vector(vector)
    for preset in UNIT_PRESETS:
        if all(abs(a - b) < 1e-9 for a, b in zip(preset.vector, data)):
            return preset
    return None


class UnitDialog(QDialog):
    """Manually edit the 7 exponents of a dimension vector."""

    def __init__(self, vector=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自定义量纲")
        self.setModal(True)
        self.setMinimumWidth(460)
        self._spins: list[QDoubleSpinBox] = []
        initial = unit_vector(vector)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(12)
        lay.addWidget(section_title("自定义量纲", "为变量设置 7 个基本物理量的指数"))
        hint = label(
            "量纲向量顺序：L 长度 · M 质量 · T 时间 · I 电流 · Θ 温度 · N 物质的量 · J 光强。"
            "指数可以为负数或小数。",
            "Caption",
            wrap=True,
        )
        lay.addWidget(hint)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        for i, (symbol, name) in enumerate(DIMENSIONS):
            grid.addWidget(label(f"{symbol} · {name}", "Caption"), i // 4, (i % 4) * 2)
            spin = QDoubleSpinBox()
            spin.setRange(-9.0, 9.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setValue(initial[i])
            spin.valueChanged.connect(self._refresh_preview)
            self._spins.append(spin)
            grid.addWidget(spin, i // 4, (i % 4) * 2 + 1)
        lay.addLayout(grid)
        self.preview = label("", "CardTitle")
        lay.addWidget(self.preview)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("取消")
        cancel.setProperty("variant", "secondary")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        ok = QPushButton("使用这个量纲")
        ok.setProperty("variant", "primary")
        ok.clicked.connect(self.accept)
        buttons.addWidget(ok)
        lay.addLayout(buttons)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        self.preview.setText("量纲：" + format_units(self.vector()))

    def vector(self) -> list[float]:
        return [float(spin.value()) for spin in self._spins]


class UnitComboBox(QComboBox):
    unitChanged = Signal(list)

    def __init__(self, vector=None, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(150)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.setStyleSheet(
            "QComboBox { background:#F2F2F7; border:1px solid #E5E5EA; border-radius:9px;"
            "padding:7px 9px; }"
            "QComboBox:focus { background:white; border:1px solid #007AFF; }"
        )
        self._vector = unit_vector(vector)
        self._last_index = 0
        self._loading = False
        for preset in UNIT_PRESETS:
            self.addItem(f"{preset.label}", list(preset.vector))
        self.insertSeparator(self.count())
        self.addItem("自定义量纲…", "custom")
        self.set_vector(self._vector)
        self.currentIndexChanged.connect(self._on_index_changed)

    def _on_index_changed(self, index: int) -> None:
        if self._loading:
            return
        data = self.itemData(index)
        if data == "custom":
            dialog = UnitDialog(self._vector, self)
            if dialog.exec() == QDialog.Accepted:
                self.set_vector(dialog.vector())
                self.unitChanged.emit(self.current_vector())
            else:
                self._restore_last()
            return
        self._vector = unit_vector(data)
        self._last_index = index
        self.unitChanged.emit(self.current_vector())

    def _restore_last(self) -> None:
        self._loading = True
        try:
            self.set_vector(self._vector)
        finally:
            self._loading = False

    def current_vector(self) -> list[float]:
        return list(self._vector)

    def set_vector(self, vector) -> None:
        data = unit_vector(vector)
        self._vector = data
        preset = find_preset(data)
        self._loading = True
        try:
            if preset is not None:
                for i in range(self.count()):
                    item = self.itemData(i)
                    if isinstance(item, list) and all(
                        abs(a - b) < 1e-9 for a, b in zip(item, data)
                    ):
                        self.setCurrentIndex(i)
                        self._last_index = i
                        break
            else:
                custom_label = f"自定义（{format_units(data)}）"
                index = -1
                for i in range(self.count()):
                    if self.itemText(i) == custom_label:
                        index = i
                        break
                if index < 0:
                    self.insertItem(self.count() - 1, custom_label, data)
                    index = self.count() - 2
                self.setCurrentIndex(index)
                self._last_index = index
        finally:
            self._loading = False


class UnitLegend(QWidget):
    """Small caption listing the dimension-vector order."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip("量纲向量顺序：L, M, T, I, Θ, N, J")
