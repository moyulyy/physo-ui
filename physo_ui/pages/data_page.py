"""Data import, preview, variable/unit selection and dataset preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import data as D
from .. import theme
from ..mathrender import MathLabel
from ..plotting import PlotCanvas, draw_scatter
from ..state import AppState
from ..units import UnitComboBox, format_units, unit_vector
from ..widgets import (
    Card,
    EmptyState,
    HLine,
    IconBadge,
    LabeledSlider,
    ScrollPage,
    SegmentedControl,
    Toggle,
    badge,
    clear_layout,
    label,
    section_title,
)


class DropZone(QFrame):
    file_dropped = Signal(str)
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(72)
        self._active = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(12)
        lay.addWidget(IconBadge("upload", theme.BLUE, theme.BLUE_SOFT, 38))
        texts = QVBoxLayout()
        texts.setSpacing(2)
        title = label("拖拽数据文件到这里", "CardTitle")
        title.setStyleSheet("font-size:14px;")
        hint = label("CSV · TSV · NPY · NPZ · XLSX", "Caption", wrap=False)
        texts.addWidget(title)
        texts.addWidget(hint)
        lay.addLayout(texts, 1)
        self.browse = QPushButton("浏览文件…")
        self.browse.setProperty("variant", "primary")
        self.browse.setStyleSheet(
            "QPushButton { background:#007AFF; color:white; border-radius:11px;"
            "padding:9px 16px; font-weight:600; }"
            "QPushButton:hover { background:#1A88FF; }"
        )
        self.browse.clicked.connect(self.clicked.emit)
        lay.addWidget(self.browse)
        self._apply_style()

    def _apply_style(self) -> None:
        border = theme.BLUE if self._active else "#CDCDD6"
        background = theme.BLUE_SOFT if self._active else "#FBFBFD"
        self.setStyleSheet(
            f"QFrame {{ border: 2px dashed {border}; border-radius: 16px; background: {background}; }}"
            "QLabel { border: none; background: transparent; }"
        )

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            self._active = True
            self._apply_style()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):  # noqa: N802
        self._active = False
        self._apply_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        self._active = False
        self._apply_style()
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()


class FeatureRow(QWidget):
    changed = Signal()

    def __init__(self, column: str, model_name: str, vector, checked: bool, parent=None):
        super().__init__(parent)
        self.column = column
        self.model_name = model_name
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 7, 4, 7)
        lay.setSpacing(10)
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        self.checkbox.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.checkbox)
        texts = QVBoxLayout()
        texts.setSpacing(1)
        name = label(column, "CardTitle")
        name.setStyleSheet("font-size:13px;")
        self.detail = label(f"模型变量 {model_name} · {format_units(vector)}", "Caption")
        texts.addWidget(name)
        texts.addWidget(self.detail)
        lay.addLayout(texts, 1)
        self.unit = UnitComboBox(vector)
        self.unit.setFixedWidth(158)
        lay.addWidget(self.unit, alignment=Qt.AlignVCenter)
        self.checkbox.toggled.connect(lambda _: self.changed.emit())
        self.unit.unitChanged.connect(self._on_unit)

    def _on_unit(self, vector) -> None:
        self.detail.setText(f"模型变量 {self.model_name} · {format_units(vector)}")
        self.changed.emit()

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def vector(self) -> list[float]:
        return self.unit.current_vector()


class MiniToggle(QWidget):
    """Compact title + switch used to keep the modeling card short."""

    def __init__(self, title: str, tooltip: str = "", *, checked: bool = False, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lbl = label(title, "CardTitle")
        lbl.setStyleSheet("font-size:12px;")
        lbl.setToolTip(tooltip)
        lay.addWidget(lbl)
        lay.addStretch(1)
        self.toggle = Toggle(checked=checked)
        self.toggle.setToolTip(tooltip)
        lay.addWidget(self.toggle)
        self.setToolTip(tooltip)

    def isChecked(self) -> bool:  # noqa: N802
        return self.toggle.isChecked()


class DataPage(ScrollPage):
    continue_requested = Signal()
    settings_requested = Signal()

    def __init__(self, state: AppState, toast, parent=None):
        super().__init__(
            "数据",
            "导入实验数据，为每个特征手动指定量纲；可选择启用量纲分析。",
            parent,
        )
        self.state = state
        self.toast = toast
        self._library = D.dataset_library()
        self._meta: dict = {}
        self._unit_by_col: dict[str, list[float]] = {}
        self._checked: set[str] = set()
        self._rows: dict[str, FeatureRow] = {}
        self._build()
        self.state.raw_changed.connect(self._on_raw_changed)
        self.state.config_changed.connect(self._update_constants_note)
        self._on_raw_changed()
        self._update_constants_note()

    # ------------------------------------------------------------------ UI #
    def _build(self) -> None:
        self.layout().setContentsMargins(28, 20, 28, 0)
        self.layout().setSpacing(12)
        self.body_layout.setContentsMargins(0, 2, 6, 14)
        self.body_layout.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(14)
        top.addWidget(self._build_source_card(), 2)
        self.demo_card = self._build_demo_card()
        top.addWidget(self.demo_card, 1)
        top_host = QWidget()
        top_host.setLayout(top)
        top_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.body_layout.addWidget(top_host)

        main = QHBoxLayout()
        main.setSpacing(14)
        main.addWidget(self._build_preview_card(), 1)
        self.setup_card = self._build_setup_card()
        self.setup_card.setMinimumWidth(430)
        self.setup_card.setMaximumWidth(470)
        main.addWidget(self.setup_card, 0)
        self.body_layout.addLayout(main)
        self.body_layout.addStretch(1)

    def _build_source_card(self) -> Card:
        card = Card(padding=14)
        card.layout.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(IconBadge("folder", theme.BLUE, theme.BLUE_SOFT, 32))
        head.addWidget(section_title("数据来源", "本地文件 · 拖拽导入"), 1)
        self.source_badge = badge("未导入", "gray")
        head.addWidget(self.source_badge)
        card.layout.addLayout(head)

        self.drop = DropZone()
        self.drop.clicked.connect(self._choose_file)
        self.drop.file_dropped.connect(self._load_path)
        card.layout.addWidget(self.drop)

        self.file_title = label("尚未导入数据", "CardTitle")
        self.file_title.setStyleSheet("font-size:14px;")
        self.file_meta = label("导入后会显示文件信息", "Caption", wrap=False)
        card.layout.addWidget(self.file_title)
        card.layout.addWidget(self.file_meta)
        card.layout.addStretch(1)
        return card

    def _build_demo_card(self) -> Card:
        card = Card(padding=14)
        card.layout.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(section_title("示例数据集", "内置物理数据"), 1)
        open_btn = QPushButton("打开")
        open_btn.setProperty("variant", "link")
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(D.DATASETS_DIR)))
        )
        head.addWidget(open_btn)
        card.layout.addLayout(head)

        self.demo_combo = QComboBox()
        for file, meta in self._library.items():
            self.demo_combo.addItem(meta.get("name", file), file)
        self.demo_combo.currentIndexChanged.connect(self._refresh_demo)
        card.layout.addWidget(self.demo_combo)

        formula_box = QFrame()
        formula_box.setObjectName("InnerCard")
        f_lay = QVBoxLayout(formula_box)
        f_lay.setContentsMargins(6, 2, 6, 2)
        self.demo_formula = MathLabel("", fontsize=12, min_height=26)
        self.demo_formula.setFixedHeight(44)
        f_lay.addWidget(self.demo_formula)
        card.layout.addWidget(formula_box)

        self.demo_desc = label("", "Caption", wrap=False)
        self.demo_units = label("", "Caption", wrap=False)
        card.layout.addStretch(1)
        self.load_demo_btn = QPushButton("载入示例  →")
        self.load_demo_btn.setProperty("variant", "secondary")
        self.load_demo_btn.clicked.connect(self._load_selected_demo)
        card.layout.addWidget(self.load_demo_btn)
        self._refresh_demo()
        return card

    def _build_preview_card(self) -> Card:
        card = Card(padding=14)
        card.layout.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(section_title("数据预览", "前 200 行"), 1)
        self.rows_badge = badge("— 行", "gray")
        self.cols_badge = badge("— 列", "gray")
        self.missing_badge = badge("缺失 —", "gray")
        head.addWidget(self.rows_badge)
        head.addWidget(self.cols_badge)
        head.addWidget(self.missing_badge)
        self.preview_segment = SegmentedControl(["表格", "散点图"], current=0)
        self.preview_segment.currentChanged.connect(self._on_preview_mode)
        head.addWidget(self.preview_segment)
        self.plot_x_combo = QComboBox()
        self.plot_x_combo.setMinimumWidth(104)
        self.plot_x_combo.currentIndexChanged.connect(self._draw_preview)
        head.addWidget(self.plot_x_combo)
        card.layout.addLayout(head)

        self.preview_stack = QStackedWidget()
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.horizontalHeader().setFixedHeight(30)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.table.setShowGrid(False)
        self.table.setMinimumHeight(300)
        self.table.setMaximumHeight(320)
        self.preview_stack.addWidget(self.table)

        chart_page = QWidget()
        chart_lay = QVBoxLayout(chart_page)
        chart_lay.setContentsMargins(0, 0, 0, 0)
        self.preview_plot = PlotCanvas(5.8, 3.0)
        self.preview_plot.clear_message("导入数据后显示散点图")
        chart_lay.addWidget(self.preview_plot)
        self.preview_stack.addWidget(chart_page)
        card.layout.addWidget(self.preview_stack)

        self.preview_empty = EmptyState(
            "还没有数据", "上传文件，或从上方选择一个内置示例。", "data"
        )
        self.preview_empty.setVisible(False)
        card.layout.addWidget(self.preview_empty)
        return card

    def _build_setup_card(self) -> Card:
        card = Card(padding=8)
        card.layout.setSpacing(4)
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(label("建模设置", "SectionTitle"), 1)
        self.apply_btn = QPushButton("应用到建模  →")
        self.apply_btn.setProperty("variant", "primary")
        self.apply_btn.setMinimumHeight(32)
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        header.addWidget(self.apply_btn)
        card.layout.addLayout(header)

        self.formula_preview = label("", "Caption", wrap=True)
        self.formula_preview.setStyleSheet(
            f"background:{theme.CARD_ALT}; border:1px solid {theme.SEPARATOR}; border-radius:9px;"
            "padding:7px 9px; font-family:Consolas, 'Cascadia Code', monospace; font-size:11px;"
        )
        self.formula_preview.setMaximumHeight(38)
        self.formula_preview.setVisible(False)

        target_bar = QHBoxLayout()
        target_bar.setSpacing(6)
        target_bar.addWidget(label("目标表达式变量 y", "CardTitle"))
        target_bar.addWidget(label("被拟合列，不参与特征", "Caption"))
        target_bar.addStretch(1)
        self.reset_btn = QPushButton("恢复推荐")
        self.reset_btn.setProperty("variant", "link")
        self.reset_btn.setToolTip("恢复示例数据推荐的 y、特征与量纲")
        self.reset_btn.clicked.connect(self._reset_recommended)
        target_bar.addWidget(self.reset_btn)
        target_bar.addWidget(label("量纲", "Caption"))
        card.layout.addLayout(target_bar)

        y_row = QHBoxLayout()
        y_row.setSpacing(8)
        self.y_combo = QComboBox()
        self.y_combo.currentIndexChanged.connect(self._on_y_changed)
        y_row.addWidget(self.y_combo, 1)
        self.y_unit_combo = UnitComboBox([0.0] * 7)
        self.y_unit_combo.setFixedWidth(158)
        self.y_unit_combo.unitChanged.connect(self._on_y_unit_changed)
        y_row.addWidget(self.y_unit_combo)
        card.layout.addLayout(y_row)

        card.layout.addWidget(HLine())
        x_head = QHBoxLayout()
        x_head.setSpacing(6)
        x_head.addWidget(label("参与表达式的特征 X", "CardTitle"))
        x_head.addWidget(label("（多选，逐行设量纲）", "Caption"))
        x_head.addStretch(1)
        all_btn = QPushButton("全选")
        all_btn.setProperty("variant", "link")
        all_btn.clicked.connect(self._select_all_x)
        none_btn = QPushButton("清空")
        none_btn.setProperty("variant", "link")
        none_btn.clicked.connect(self._clear_x)
        x_head.addWidget(all_btn)
        x_head.addWidget(none_btn)
        card.layout.addLayout(x_head)

        feature_columns = QHBoxLayout()
        feature_columns.setContentsMargins(4, 0, 4, 0)
        feature_columns.setSpacing(8)
        participate = label("参与", "Caption")
        participate.setFixedWidth(28)
        feature_columns.addWidget(participate)
        feature_columns.addWidget(label("特征 / 模型变量", "Caption"), 1)
        unit_col = label("量纲", "Caption")
        unit_col.setFixedWidth(158)
        unit_col.setAlignment(Qt.AlignCenter)
        feature_columns.addWidget(unit_col)
        card.layout.addLayout(feature_columns)

        self.feature_scroll = QScrollArea()
        self.feature_scroll.setWidgetResizable(True)
        self.feature_scroll.setFrameShape(QFrame.NoFrame)
        self.feature_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.feature_scroll.setMaximumHeight(84)
        self.feature_scroll.setMinimumHeight(58)
        self.feature_host = QWidget()
        self.feature_layout = QVBoxLayout(self.feature_host)
        self.feature_layout.setContentsMargins(0, 0, 3, 0)
        self.feature_layout.setSpacing(0)
        self.feature_scroll.setWidget(self.feature_host)
        card.layout.addWidget(self.feature_scroll)
        self.feature_hint = label("", "Caption", wrap=True)
        self.feature_hint.setVisible(False)

        constants_row = QHBoxLayout()
        constants_row.setSpacing(6)
        self.constants_note = label("", "Caption", wrap=True)
        constants_row.addWidget(self.constants_note, 1)
        constants_btn = QPushButton("设置常量")
        constants_btn.setProperty("variant", "link")
        constants_btn.setToolTip("在模型页添加无量纲常量（如 π、e）")
        constants_btn.clicked.connect(self.settings_requested.emit)
        constants_row.addWidget(constants_btn)
        card.layout.addLayout(constants_row)

        card.layout.addWidget(HLine())
        toggles = QHBoxLayout()
        toggles.setSpacing(10)
        self.split_toggle = MiniToggle(
            "留出评估集", "按比例拆分训练集/评估集；评估集不参与训练", checked=True
        )
        self.split_toggle.toggle.toggled.connect(self._on_split_toggled)
        self.norm_toggle = MiniToggle(
            "标准化", "缩放到均值 0/方差 1；结果页会换算回原始量纲", checked=False
        )
        self.norm_toggle.toggle.toggled.connect(self._on_selection_changed)
        self.da_toggle = MiniToggle(
            "量纲分析", "用你设置的单位约束表达式；自由常数按无量纲处理", checked=False
        )
        self.da_toggle.toggle.toggled.connect(self._on_selection_changed)
        for widget in (self.split_toggle, self.norm_toggle, self.da_toggle):
            toggles.addWidget(widget, 1)
        card.layout.addLayout(toggles)

        self.split_slider = LabeledSlider("训练集比例", 50, 90, 80, suffix=" %", step=5)
        self.split_slider.valueChanged.connect(self._on_selection_changed)
        card.layout.addWidget(self.split_slider)

        self.selection_info = label("", "Caption", wrap=True)
        self.selection_info.setVisible(False)

        return card

    # -------------------------------------------------------------- loading #
    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据文件",
            str(D.DATASETS_DIR),
            "数据文件 (*.csv *.tsv *.txt *.dat *.npy *.npz *.xlsx *.xls);;所有文件 (*.*)",
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            raw = D.load_table(path)
        except Exception as exc:
            QMessageBox.warning(self, "无法导入数据", D.friendly_error(exc))
            return
        self._meta = {}
        self.state.set_raw(raw)
        self.toast(f"已导入 {raw.name}", "success")

    def _load_selected_demo(self) -> None:
        file = self.demo_combo.currentData()
        if not file:
            return
        try:
            raw = D.load_bundled(file)
        except Exception as exc:
            QMessageBox.warning(self, "无法载入示例", D.friendly_error(exc))
            return
        self._meta = self._library.get(file, {})
        self.state.set_raw(raw)
        self.toast(f"已载入示例：{raw.name}", "success")

    def _refresh_demo(self, *_args) -> None:
        file = self.demo_combo.currentData()
        meta = self._library.get(file, {})
        formula = meta.get("formula_latex") or meta.get("formula") or ""
        self.demo_formula.set_latex(formula, fallback=meta.get("formula", ""))
        self.demo_desc.setText(meta.get("description", ""))
        self.demo_combo.setToolTip(meta.get("description", ""))
        x_units = meta.get("x_units", [])
        x_cols = meta.get("x_columns", [])
        y_units = meta.get("y_units", [])
        parts = [f"{col} [{format_units(vec)}]" for col, vec in zip(x_cols, x_units)]
        if meta.get("y_column"):
            parts.append(f"{meta['y_column']} [{format_units(y_units)}]")
        self.demo_units.setText(" · ".join(parts))
        self.load_demo_btn.setEnabled(bool(file))

    # ---------------------------------------------------------------- state #
    def _meta_for(self, raw: D.RawDataset) -> dict:
        if self._meta:
            return self._meta
        library = self._library
        if raw.path:
            name = Path(raw.path).name
            if name in library:
                return library[name]
        for meta in library.values():
            if meta.get("name") == raw.name:
                return meta
        return {}

    def _on_raw_changed(self) -> None:
        raw = self.state.raw
        if raw is None:
            self._reset_view()
            return
        meta = self._meta_for(raw)
        self.file_title.setText(raw.name)
        file_name = Path(raw.path).name if raw.path else raw.name
        self.file_meta.setText(f"{raw.n_rows} 行 · {raw.n_cols} 列 · {file_name}")
        self.file_meta.setToolTip(raw.path or "")
        self.reset_btn.setEnabled(bool(meta))
        self.source_badge.setText("已导入")
        self.source_badge.setObjectName("BadgeGreen")
        _repolish(self.source_badge)
        if meta.get("file"):
            demo_index = self.demo_combo.findData(meta["file"])
            if demo_index >= 0 and demo_index != self.demo_combo.currentIndex():
                self.demo_combo.setCurrentIndex(demo_index)
        self._fill_table(raw.frame)
        self.preview_stack.setVisible(True)
        self.preview_empty.setVisible(False)
        self.preview_segment.setEnabled(True)
        self.preview_stack.setCurrentIndex(0)
        missing = int(raw.frame.isna().sum().sum())
        self.rows_badge.setText(f"{raw.n_rows} 行")
        self.cols_badge.setText(f"{raw.n_cols} 列")
        self.missing_badge.setText(f"缺失 {missing}")

        numeric = self._numeric_columns(raw.frame)
        self._unit_by_col = {col: [0.0] * 7 for col in numeric}
        for col, vec in zip(meta.get("x_columns", []), meta.get("x_units", [])):
            if col in self._unit_by_col:
                self._unit_by_col[col] = unit_vector(vec)
        if meta.get("y_column") in self._unit_by_col:
            self._unit_by_col[meta["y_column"]] = unit_vector(meta.get("y_units"))

        self.y_combo.blockSignals(True)
        self.y_combo.clear()
        for col in numeric:
            self.y_combo.addItem(str(col))
        if numeric:
            default_y = meta.get("y_column")
            index = numeric.index(default_y) if default_y in numeric else len(numeric) - 1
            self.y_combo.setCurrentIndex(index)
        self.y_combo.blockSignals(False)

        y = self.y_combo.currentText()
        self._checked = {col for col in numeric if col != y}
        if y:
            self.y_unit_combo.set_vector(self._unit_by_col.get(y, [0.0] * 7))
        self._rebuild_features()
        self._update_plot_combo()
        self._update_info()
        self._draw_preview()
        self.apply_btn.setEnabled(bool(numeric))

    def _reset_view(self) -> None:
        self.file_title.setText("尚未导入数据")
        self.file_meta.setText("导入后会显示文件信息")
        self.file_meta.setToolTip("")
        self.source_badge.setText("未导入")
        self.source_badge.setObjectName("BadgeGray")
        _repolish(self.source_badge)
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.preview_stack.setVisible(False)
        self.preview_empty.setVisible(True)
        self.preview_segment.setEnabled(False)
        self.formula_preview.setText("")
        self.constants_note.setText("")
        self.reset_btn.setEnabled(False)
        self.y_combo.clear()
        clear_layout(self.feature_layout)
        self._rows.clear()
        self._checked.clear()
        self.rows_badge.setText("— 行")
        self.cols_badge.setText("— 列")
        self.missing_badge.setText("缺失 —")
        self.plot_x_combo.clear()
        self.preview_plot.clear_message("导入数据后显示散点图")
        self.selection_info.setText("")
        self.feature_hint.setText("")
        self.apply_btn.setEnabled(False)

    def _fill_table(self, frame: pd.DataFrame) -> None:
        preview = frame.head(200)
        self.table.clear()
        self.table.setColumnCount(len(preview.columns) + 1)
        self.table.setRowCount(len(preview))
        self.table.setHorizontalHeaderLabels(["#", *[str(c) for c in preview.columns]])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, self.table.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 42)
        for i in range(self.table.columnCount()):
            item = self.table.horizontalHeaderItem(i)
            if item is not None:
                item.setTextAlignment(Qt.AlignCenter)
        for r in range(len(preview)):
            index_item = QTableWidgetItem(str(r + 1))
            index_item.setFlags(Qt.ItemIsEnabled)
            index_item.setForeground(QColor(theme.GRAY))
            index_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 0, index_item)
            for c, col in enumerate(preview.columns):
                value = preview.iloc[r, c]
                if pd.isna(value):
                    text = "—"
                elif isinstance(value, (float, np.floating)):
                    text = f"{value:.6g}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c + 1, item)

    def _numeric_columns(self, frame: pd.DataFrame) -> list[str]:
        cols = []
        for col in frame.columns:
            converted = pd.to_numeric(frame[col], errors="coerce")
            if converted.notna().sum() >= max(2, int(len(frame) * 0.5)):
                cols.append(col)
        return cols

    def _rebuild_features(self) -> None:
        clear_layout(self.feature_layout)
        self._rows.clear()
        y = self.y_combo.currentText()
        numeric = self._numeric_columns(self.state.raw.frame) if self.state.raw is not None else []
        columns = [col for col in numeric if col != y]
        if not columns:
            self.feature_layout.addWidget(label("没有可作为输入特征的数值列。", "Caption"))
            self.feature_layout.addStretch(1)
            self.feature_hint.setText("请修改目标变量，或检查文件是否为数值列。")
            return
        mapping = D.sanitize_names([*columns, y]) if y else D.sanitize_names(columns)
        for col in columns:
            row = FeatureRow(
                col,
                mapping.get(col, col),
                self._unit_by_col.get(col, [0.0] * 7),
                col in self._checked,
            )
            row.changed.connect(self._on_selection_changed)
            row.unit.unitChanged.connect(lambda vec, c=col: self._set_unit(c, vec))
            self.feature_layout.addWidget(row)
            self._rows[col] = row
        self.feature_layout.addStretch(1)
        self.feature_hint.setText(
            f"{len(columns)} 个可用特征 · 勾选后参与符号回归；量纲可逐个选择或自定义。"
        )

    def _set_unit(self, column: str, vector) -> None:
        self._unit_by_col[column] = unit_vector(vector)
        self._update_info()

    def _on_y_unit_changed(self, vector) -> None:
        y = self.y_combo.currentText()
        if y:
            self._unit_by_col[y] = unit_vector(vector)
        self._update_info()

    def _on_y_changed(self, *_args) -> None:
        y = self.y_combo.currentText()
        self._checked.discard(y)
        if y:
            self.y_unit_combo.set_vector(self._unit_by_col.get(y, [0.0] * 7))
        self._rebuild_features()
        self._update_plot_combo()
        self._update_info()
        self._draw_preview()

    def _select_all_x(self) -> None:
        for row in self._rows.values():
            row.checkbox.setChecked(True)

    def _clear_x(self) -> None:
        for row in self._rows.values():
            row.checkbox.setChecked(False)

    def _selected_x(self) -> list[str]:
        return [col for col, row in self._rows.items() if row.is_checked()]

    def _on_selection_changed(self, *_args) -> None:
        self._checked = set(self._selected_x())
        self._update_info()
        self._update_plot_combo()
        self._draw_preview()

    def _on_split_toggled(self, checked: bool) -> None:
        self.split_slider.setEnabled(checked)
        self._update_info()

    def _update_info(self) -> None:
        x = self._selected_x()
        y = self.y_combo.currentText()
        y_unit = self._unit_by_col.get(y, [0.0] * 7)
        x_parts = [
            f"{col} [{format_units(self._unit_by_col.get(col, [0.0] * 7))}]" for col in x
        ]
        if y and x:
            formula = f"{y} [{format_units(y_unit)}] = f( {', '.join(x_parts)} )"
        elif y:
            formula = f"{y} [{format_units(y_unit)}] = f( ? )"
        else:
            formula = "请先导入数据并选择目标变量"
        self.formula_preview.setText(formula)
        self.feature_hint.setText(
            f"{formula}  ·  已选 {len(x)} / {len(self._rows)} 个特征"
            if self._rows
            else "没有可作为输入特征的数值列"
        )
        self.feature_hint.setVisible(bool(self._rows))
        self.apply_btn.setEnabled(bool(x and y and self.state.raw is not None and self._rows))
        self.split_slider.setEnabled(self.split_toggle.isChecked())

    def _on_preview_mode(self, index: int) -> None:
        self.preview_stack.setCurrentIndex(index)
        if index == 1:
            self._draw_preview()

    def _reset_recommended(self) -> None:
        raw = self.state.raw
        if raw is None:
            return
        meta = self._meta_for(raw)
        numeric = self._numeric_columns(raw.frame)
        if not numeric:
            return
        y = meta.get("y_column")
        if y not in numeric:
            y = numeric[-1]
        units = {col: [0.0] * 7 for col in numeric}
        for col, vec in zip(meta.get("x_columns", []), meta.get("x_units", [])):
            if col in units:
                units[col] = unit_vector(vec)
        if y in units:
            units[y] = unit_vector(meta.get("y_units"))
        recommended_x = [col for col in meta.get("x_columns", []) if col in numeric and col != y]
        if not recommended_x:
            recommended_x = [col for col in numeric if col != y]
        self._unit_by_col = units
        self._checked = set(recommended_x)
        self.y_combo.blockSignals(True)
        self.y_combo.setCurrentText(str(y))
        self.y_combo.blockSignals(False)
        self.y_unit_combo.set_vector(units.get(y, [0.0] * 7))
        self._rebuild_features()
        self._update_plot_combo()
        self._update_info()
        self._draw_preview()
        self.toast("已恢复示例推荐的变量与量纲。", "info")

    def _update_constants_note(self) -> None:
        cfg = self.state.config
        free = [f"c{i}" for i in range(max(0, int(cfg.n_free_consts)))]
        fixed = [f"{v:g}" for v in cfg.fixed_consts]
        if free or fixed:
            parts = []
            if free:
                parts.append("自由 " + ", ".join(free))
            if fixed:
                parts.append("固定 " + ", ".join(fixed))
            self.constants_note.setText("表达式无量纲常量：" + " · ".join(parts))
        else:
            self.constants_note.setText("表达式暂未启用常量；可点「设置常量」添加 π、e 等无量纲常量。")

    def _update_plot_combo(self) -> None:
        current = self.plot_x_combo.currentText()
        x = self._selected_x()
        self.plot_x_combo.blockSignals(True)
        self.plot_x_combo.clear()
        for col in x:
            self.plot_x_combo.addItem(col)
        if current in x:
            self.plot_x_combo.setCurrentText(current)
        self.plot_x_combo.blockSignals(False)

    def _draw_preview(self, *_args) -> None:
        raw = self.state.raw
        x_cols = self._selected_x()
        if raw is None or not x_cols:
            self.preview_plot.clear_message("勾选特征后显示散点图")
            return
        x_col = self.plot_x_combo.currentText() or x_cols[0]
        y_col = self.y_combo.currentText()
        if x_col not in raw.frame.columns or y_col not in raw.frame.columns:
            return
        x = pd.to_numeric(raw.frame[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(raw.frame[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            self.preview_plot.clear_message("有效数据点太少")
            return
        draw_scatter(self.preview_plot, x[mask], y[mask], x_col, y_col)

    # ---------------------------------------------------------------- apply #
    def _apply(self) -> None:
        raw = self.state.raw
        if raw is None:
            self.toast("请先导入数据。", "error")
            return
        x = self._selected_x()
        y = self.y_combo.currentText()
        if not x:
            self.toast("请至少选择一个输入特征。", "error")
            return
        if not y:
            self.toast("请选择目标变量。", "error")
            return
        x_units = [self._unit_by_col.get(col, [0.0] * 7) for col in x]
        y_units = self._unit_by_col.get(y, [0.0] * 7)
        try:
            prepared = D.prepare_data(
                raw,
                x,
                y,
                train_fraction=self.split_slider.value() / 100.0,
                split_enabled=self.split_toggle.isChecked(),
                normalize=self.norm_toggle.isChecked(),
                x_units=x_units,
                y_units=y_units,
                use_units=bool(self.da_toggle.isChecked() and not self.norm_toggle.isChecked()),
            )
        except Exception as exc:
            QMessageBox.warning(self, "无法准备数据", D.friendly_error(exc))
            return
        if prepared.n_samples < 30:
            self.toast("有效样本少于 30 行，训练结果可能不稳定。", "warn")
        if self.da_toggle.isChecked() and not self.norm_toggle.isChecked() and not prepared.has_units:
            self.toast("所有变量都是无量纲，量纲分析不会产生额外约束。", "info")
        if self.da_toggle.isChecked() and self.norm_toggle.isChecked():
            self.toast("标准化后变量按无量纲处理，本次训练已关闭量纲分析。", "warn")
        self.state.set_prepared(prepared)
        self.toast(f"已准备 {prepared.describe()}", "success")
        self.continue_requested.emit()


def _repolish(widget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
