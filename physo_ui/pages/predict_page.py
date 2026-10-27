"""Inference page: single-point and batch prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import data as D
from .. import engine as E
from .. import theme
from ..mathrender import MathLabel
from ..state import AppState
from ..units import format_units
from ..widgets import (
    Card,
    EmptyState,
    ScrollPage,
    SegmentedControl,
    badge,
    clear_layout,
    label,
    section_title,
)


class PredictPage(ScrollPage):
    train_requested = Signal()
    result_requested = Signal()

    def __init__(self, state: AppState, toast, parent=None):
        super().__init__("推理", "用发现的关系式计算新样本，并导出预测结果。", parent)
        self.state = state
        self.toast = toast
        self._single_values: dict[str, QDoubleSpinBox] = {}
        self._history: list[dict] = []
        self._batch_result: pd.DataFrame | None = None
        self._batch_source: pd.DataFrame | None = None
        self._build()
        self.state.result_changed.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ UI #
    def _build(self) -> None:
        outer = self.body_layout

        self.empty = EmptyState(
            "还没有可用于推理的模型",
            "完成训练或载入一个已保存的 result.json。",
            "predict",
        )
        row = QHBoxLayout()
        row.addStretch(1)
        load_btn = QPushButton("载入结果…")
        load_btn.setProperty("variant", "secondary")
        load_btn.clicked.connect(self.load_result)
        row.addWidget(load_btn)
        go_btn = QPushButton("前往训练  →")
        go_btn.setProperty("variant", "primary")
        go_btn.clicked.connect(self.train_requested.emit)
        row.addWidget(go_btn)
        row.addStretch(1)
        self.empty.add_action_layout(row)
        outer.addWidget(self.empty)

        self.content = QWidget()
        lay = QVBoxLayout(self.content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        outer.addWidget(self.content)
        self._build_model_card(lay)

        self.segment = SegmentedControl(["单点输入", "批量文件"], current=0)
        self.segment.currentChanged.connect(self._on_mode)
        lay.addWidget(self.segment, alignment=Qt.AlignLeft)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_single_panel())
        self.stack.addWidget(self._build_batch_panel())
        lay.addWidget(self.stack)
        lay.addStretch(1)

    def _build_model_card(self, parent_layout) -> None:
        card = Card(padding=14)
        self.model_card = card
        card.layout.setSpacing(5)
        card.setMaximumHeight(188)
        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(section_title("当前模型", "推理使用的表达式"), 1)
        self.switch_btn = QPushButton("更换公式")
        self.switch_btn.setProperty("variant", "link")
        self.switch_btn.clicked.connect(self.result_requested.emit)
        head.addWidget(self.switch_btn)
        card.layout.addLayout(head)
        self.expression_math = MathLabel("", fontsize=20, min_height=40)
        self.expression_math.setMaximumHeight(58)
        self.expression_math.clear_formula("—")
        card.layout.addWidget(self.expression_math)
        self.expression_label = label("", "Caption", wrap=True, selectable=True)
        self.expression_label.setStyleSheet(
            "font-family: Consolas, 'Cascadia Code', monospace; color:#8E8E93;"
        )
        self.expression_label.setMaximumHeight(0)
        self.expression_label.setVisible(False)
        self.model_units = label("", "Caption", wrap=True)
        self.model_units.setMaximumHeight(18)
        card.layout.addWidget(self.model_units)
        meta = QHBoxLayout()
        meta.setSpacing(8)
        self.model_r2 = badge("R² —", "green")
        self.model_rmse = badge("RMSE —", "gray")
        self.model_vars = badge("输入 —", "gray")
        self.model_source = badge("—", "blue")
        meta.addWidget(self.model_r2)
        meta.addWidget(self.model_rmse)
        meta.addWidget(self.model_vars)
        meta.addWidget(self.model_source)
        meta.addStretch(1)
        card.layout.addLayout(meta)
        parent_layout.addWidget(card)

    def _build_single_panel(self) -> QWidget:
        panel = QWidget()
        lay = QHBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        input_card = Card(padding=18)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(section_title("输入变量", "按原始量纲填写"), 1)
        calc = QPushButton("计算预测  →")
        calc.setProperty("variant", "primary")
        calc.setMinimumHeight(32)
        calc.clicked.connect(self._predict_single)
        head.addWidget(calc)
        self.mean_btn = QPushButton("填入均值")
        self.mean_btn.setProperty("variant", "link")
        self.mean_btn.clicked.connect(self._fill_means)
        head.addWidget(self.mean_btn)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.setProperty("variant", "link")
        self.reset_btn.clicked.connect(self._reset_inputs)
        head.addWidget(self.reset_btn)
        input_card.layout.addLayout(head)

        self.form_host = QWidget()
        self.form_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.form_layout = QGridLayout(self.form_host)
        self.form_layout.setContentsMargins(0, 2, 0, 2)
        self.form_layout.setHorizontalSpacing(12)
        self.form_layout.setVerticalSpacing(12)
        input_card.layout.addWidget(self.form_host)
        hint = label(
            "变量名显示为原始列名；悬停单位可查看模型变量。修改数值后随时重新计算。",
            "Caption",
            wrap=True,
        )
        input_card.layout.addWidget(hint)
        input_card.layout.addStretch(1)
        lay.addWidget(input_card, 3, Qt.AlignTop)

        out_card = Card(padding=18)
        out_card.layout.addWidget(section_title("预测结果", "单点推理"))
        pred_box = QFrame()
        pred_box.setStyleSheet(
            "QFrame { background:#EAF3FF; border:1px solid #D6E8FF; border-radius:14px; }"
            "QLabel { background:transparent; }"
        )
        pv = QVBoxLayout(pred_box)
        pv.setContentsMargins(16, 12, 16, 12)
        pv.setSpacing(2)
        self.pred_caption = label("目标变量 y", "Caption")
        self.pred_label = QLabel("—")
        self.pred_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.pred_label.setStyleSheet(
            f"font-size:34px; font-weight:700; color:{theme.BLUE}; padding:0;"
        )
        self.pred_unit = label("", "Caption")
        self.pred_note = label("输入数值后点击「计算预测」。", "Caption", wrap=True)
        pv.addWidget(self.pred_caption)
        pv.addWidget(self.pred_label)
        pv.addWidget(self.pred_unit)
        pv.addSpacing(2)
        pv.addWidget(self.pred_note)
        out_card.layout.addWidget(pred_box)

        self.history_table = QTableWidget()
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setShowGrid(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setMinimumHeight(120)
        self.history_table.setMaximumHeight(180)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        out_card.layout.addWidget(self.history_table)
        self.history_empty = label("还没有单点预测记录；计算结果会显示在这里。", "Caption", wrap=True)
        out_card.layout.addWidget(self.history_empty)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.clear_hist_btn = QPushButton("清空记录")
        self.clear_hist_btn.setProperty("variant", "link")
        self.clear_hist_btn.clicked.connect(self._clear_history)
        actions.addWidget(self.clear_hist_btn)
        self.export_single_btn = QPushButton("导出预测 CSV")
        self.export_single_btn.setProperty("variant", "secondary")
        self.export_single_btn.setEnabled(False)
        self.export_single_btn.clicked.connect(self._export_single)
        actions.addWidget(self.export_single_btn)
        out_card.layout.addLayout(actions)
        lay.addWidget(out_card, 2, Qt.AlignTop)
        return panel

    def _build_batch_panel(self) -> QWidget:
        panel = QWidget()
        lay = QHBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        source_card = Card(padding=18)
        source_card.layout.addWidget(section_title("批量输入", "从文件读取与训练一致的列"))
        row = QHBoxLayout()
        row.setSpacing(8)
        choose = QPushButton("选择 CSV / NPY / NPZ")
        choose.setProperty("variant", "primary")
        choose.clicked.connect(self._choose_batch)
        row.addWidget(choose)
        use_current = QPushButton("使用当前数据集")
        use_current.setProperty("variant", "secondary")
        use_current.clicked.connect(self._use_current_dataset)
        row.addWidget(use_current)
        row.addStretch(1)
        source_card.layout.addLayout(row)
        self.batch_info = label("尚未选择批量数据。", "Caption", wrap=True)
        source_card.layout.addWidget(self.batch_info)
        self.batch_table = QTableWidget()
        self.batch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.setShowGrid(False)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.batch_table.setMinimumHeight(240)
        source_card.layout.addWidget(self.batch_table)
        self.run_batch_btn = QPushButton("批量预测  →")
        self.run_batch_btn.setProperty("variant", "primary")
        self.run_batch_btn.setMinimumHeight(38)
        self.run_batch_btn.setEnabled(False)
        self.run_batch_btn.clicked.connect(self._predict_batch)
        source_card.layout.addWidget(self.run_batch_btn)
        lay.addWidget(source_card, 3)

        out_card = Card(padding=18)
        head = QHBoxLayout()
        head.addWidget(section_title("预测输出", "包含残差（若存在目标列）"), 1)
        self.batch_badge = badge("0 行", "gray")
        head.addWidget(self.batch_badge)
        out_card.layout.addLayout(head)
        self.batch_result_table = QTableWidget()
        self.batch_result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_result_table.verticalHeader().setVisible(False)
        self.batch_result_table.setShowGrid(False)
        self.batch_result_table.setAlternatingRowColors(True)
        self.batch_result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.batch_result_table.setMinimumHeight(260)
        out_card.layout.addWidget(self.batch_result_table)
        self.batch_note = label("导入批量数据后点击「批量预测」。", "Caption", wrap=True)
        out_card.layout.addWidget(self.batch_note)
        export = QPushButton("导出批量预测 CSV")
        export.setProperty("variant", "secondary")
        self.export_batch_btn = export
        export.setEnabled(False)
        export.clicked.connect(self._export_batch)
        out_card.layout.addWidget(export)
        lay.addWidget(out_card, 2, Qt.AlignTop)
        return panel

    # -------------------------------------------------------------- refresh #
    def refresh(self) -> None:
        result = self.state.result
        self.empty.setVisible(result is None)
        self.content.setVisible(result is not None)
        if result is None:
            return
        active = result.active
        self.expression_math.set_latex(active.latex, fallback=active.display_expression)
        self.expression_label.setText(active.display_expression or active.expression)
        unit_parts = [
            f"{name} [{format_units(vec)}]" for name, vec in zip(result.x_names, result.x_units)
        ]
        unit_parts.append(f"{result.y_name} [{format_units(result.y_units)}]")
        self.model_units.setText("量纲：" + " · ".join(unit_parts))
        self.model_r2.setText(f"R² {D.format_number(active.r2_train, 6)}")
        self.model_rmse.setText(f"RMSE {D.format_number(active.rmse_train, 6)}")
        self.model_vars.setText(f"{len(result.x_names)} 个输入")
        self.model_source.setText("最佳公式" if active.source == "best" else "Pareto 公式")
        self._rebuild_form(result)
        self._history.clear()
        self._refresh_history_table()
        self.pred_label.setText("待计算")
        self.pred_label.setStyleSheet(
            f"font-size:24px; font-weight:600; color:{theme.GRAY}; padding:6px 0;"
        )
        self.pred_caption.setText(f"目标变量 {result.y_name}")
        self.pred_unit.setText(f"量纲：{format_units(result.y_units)}")
        self.pred_note.setText("输入数值后点击「计算预测」。")
        self._batch_result = None
        self.batch_result_table.clear()
        self.batch_result_table.setRowCount(0)
        self.batch_badge.setText("0 行")
        self.export_batch_btn.setEnabled(False)

    def _rebuild_form(self, result: E.RunResult) -> None:
        clear_layout(self.form_layout)
        self._single_values.clear()
        medians = result.summary.get("x_median") or [0.0] * len(result.x_names)
        mins = result.summary.get("x_min") or [None] * len(result.x_names)
        maxs = result.summary.get("x_max") or [None] * len(result.x_names)
        for i, name in enumerate(result.x_names):
            row, col = divmod(i, 2)
            title = result.x_columns[i] if i < len(result.x_columns) else name
            unit_text = format_units(result.x_units[i]) if i < len(result.x_units) else "—"
            field = QFrame()
            field.setStyleSheet(
                "QFrame { background:#FAFAFC; border:1px solid #EEEEF3; border-radius:10px; }"
                "QLabel { background:transparent; border:none; }"
            )
            fl = QVBoxLayout(field)
            fl.setContentsMargins(10, 8, 10, 8)
            fl.setSpacing(4)
            top = QHBoxLayout()
            top.setSpacing(6)
            title_label = label(title, "CardTitle")
            title_label.setStyleSheet("font-size:12px;")
            if title != name:
                title_label.setToolTip(f"模型变量名：{name}")
            top.addWidget(title_label)
            top.addStretch(1)
            top.addWidget(label(f"[{unit_text}]", "Caption"))
            fl.addLayout(top)
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e12, 1e12)
            default = float(medians[i]) if i < len(medians) and medians[i] is not None else 0.0
            spin.setValue(default if np.isfinite(default) else 0.0)
            if i < len(mins) and i < len(maxs) and mins[i] is not None and maxs[i] is not None:
                span = float(maxs[i]) - float(mins[i])
                step = abs(span) / 100.0 if span > 0 else 1.0
                spin.setSingleStep(step)
                spin.setToolTip(
                    f"数据范围：{D.format_number(mins[i], 6)} … {D.format_number(maxs[i], 6)}"
                )
            self._single_values[name] = spin
            fl.addWidget(spin)
            self.form_layout.addWidget(field, row, col)
        for c in (0, 1):
            self.form_layout.setColumnStretch(c, 1)

    def _current_X(self) -> np.ndarray:
        result = self.state.result
        if result is None:
            return np.empty((0, 1))
        return np.array([[self._single_values[name].value()] for name in result.x_names], dtype=float)

    def _predict_single(self) -> None:
        result = self.state.result
        if result is None:
            return
        try:
            pred = result.predict_array(self._current_X())
        except Exception as exc:
            QMessageBox.warning(self, "预测失败", D.friendly_error(exc))
            return
        value = float(np.asarray(pred).ravel()[0])
        if not np.isfinite(value):
            self.pred_label.setStyleSheet(
                f"font-size:30px; font-weight:700; color:{theme.RED}; padding:0;"
            )
            self.pred_label.setText("NaN")
            self.pred_note.setText("预测结果不是有限数值，请检查输入是否在表达式定义域内。")
        else:
            self.pred_label.setStyleSheet(
                f"font-size:34px; font-weight:700; color:{theme.BLUE}; padding:0;"
            )
            self.pred_label.setText(D.format_number(value, 8))
            self.pred_note.setText(f"由当前公式计算 · {result.active.display_expression}")
        row = {name: self._single_values[name].value() for name in result.x_names}
        row["prediction"] = value
        self._history.append(row)
        self._refresh_history_table()

    def _refresh_history_table(self) -> None:
        result = self.state.result
        if result is None:
            return
        headers = [result.x_columns[i] if i < len(result.x_columns) else n for i, n in enumerate(result.x_names)]
        headers.append(f"{result.y_name}（预测）")
        self.history_table.setColumnCount(len(headers))
        self.history_table.setHorizontalHeaderLabels(headers)
        self.history_table.setRowCount(len(self._history))
        for r, row in enumerate(self._history):
            for c, name in enumerate(result.x_names):
                value = row.get(name, float("nan"))
                item = QTableWidgetItem(D.format_number(value, 6))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.history_table.setItem(r, c, item)
            pred_item = QTableWidgetItem(D.format_number(row.get("prediction"), 8))
            pred_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.history_table.setItem(r, len(result.x_names), pred_item)
        self.export_single_btn.setEnabled(bool(self._history))
        self.history_table.setVisible(bool(self._history))
        self.history_empty.setVisible(not bool(self._history))
        if self._history:
            self.history_table.scrollToBottom()

    def _fill_means(self) -> None:
        result = self.state.result
        if result is None:
            return
        means = result.summary.get("x_mean") or [0.0] * len(result.x_names)
        for i, name in enumerate(result.x_names):
            if name in self._single_values and i < len(means):
                self._single_values[name].setValue(float(means[i]))

    def _reset_inputs(self) -> None:
        result = self.state.result
        if result is None:
            return
        medians = result.summary.get("x_median") or [0.0] * len(result.x_names)
        for i, name in enumerate(result.x_names):
            if name in self._single_values:
                self._single_values[name].setValue(float(medians[i]) if i < len(medians) else 0.0)

    def _clear_history(self) -> None:
        self._history.clear()
        self._refresh_history_table()

    # ---------------------------------------------------------------- batch #
    def _choose_batch(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择批量预测数据",
            str(D.DATASETS_DIR),
            "数据文件 (*.csv *.tsv *.txt *.npy *.npz *.xlsx *.xls);;所有文件 (*.*)",
        )
        if path:
            try:
                raw = D.load_table(path)
            except Exception as exc:
                QMessageBox.warning(self, "无法读取文件", D.friendly_error(exc))
                return
            self._set_batch_source(raw.frame, raw.name)

    def _use_current_dataset(self) -> None:
        if self.state.prepared is None:
            self.toast("当前没有已准备的数据集。", "warn")
            return
        self._set_batch_source(self.state.prepared.frame, self.state.prepared.name)

    def _set_batch_source(self, frame: pd.DataFrame, name: str) -> None:
        self._batch_source = frame
        self._batch_result = None
        self.batch_info.setText(f"{name} · {len(frame)} 行 · {len(frame.columns)} 列")
        _fill_preview_table(self.batch_table, frame.head(200))
        self.run_batch_btn.setEnabled(self.state.result is not None)
        self.batch_result_table.clear()
        self.batch_result_table.setRowCount(0)
        self.export_batch_btn.setEnabled(False)
        self.batch_note.setText("确认列名与训练数据一致后点击「批量预测」。")

    def _predict_batch(self) -> None:
        result = self.state.result
        if result is None:
            return
        if self._batch_source is None:
            self.toast("请先选择批量数据。", "warn")
            return
        try:
            out = result.infer_frame(self._batch_source)
        except Exception as exc:
            QMessageBox.warning(self, "批量预测失败", D.friendly_error(exc))
            return
        self._batch_result = out
        preview = out.head(500)
        _fill_preview_table(self.batch_result_table, preview)
        self.batch_badge.setText(f"{len(out)} 行")
        self.export_batch_btn.setEnabled(True)
        finite = np.isfinite(pd.to_numeric(out["prediction"], errors="coerce").to_numpy(dtype=float))
        bad = int((~finite).sum())
        msg = f"完成 {len(out)} 行预测。"
        if bad:
            msg += f" 其中 {bad} 行结果不是有限值，请检查输入范围。"
        if "residual" in out.columns:
            metrics = D.regression_metrics(
                pd.to_numeric(out[result.y_column], errors="coerce").to_numpy(dtype=float),
                out["prediction"].to_numpy(dtype=float),
            )
            msg += f" R² = {D.format_number(metrics['r2'], 5)} · RMSE = {D.format_number(metrics['rmse'], 5)}。"
        self.batch_note.setText(msg)
        self.toast("批量预测完成。", "success")

    def _export_single(self) -> None:
        result = self.state.result
        if result is None or not self._history:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出预测结果", str(E.RUNS_DIR / "single_predictions.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        frame = pd.DataFrame(self._history)
        frame.rename(columns={name: (result.x_columns[i] if i < len(result.x_columns) else name) for i, name in enumerate(result.x_names)}, inplace=True)
        frame.rename(columns={"prediction": result.y_column + "_prediction"}, inplace=True)
        try:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", D.friendly_error(exc))
            return
        self.toast(f"已导出 {path}", "success")

    def _export_batch(self) -> None:
        if self._batch_result is None or self._batch_result.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出批量预测", str(E.RUNS_DIR / "batch_predictions.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._batch_result.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", D.friendly_error(exc))
            return
        self.toast(f"已导出 {path}", "success")

    def _on_mode(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    def load_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "载入结果", str(E.RUNS_DIR), "结果文件 (result.json *.json);;所有文件 (*.*)"
        )
        if not path:
            return
        try:
            result = E.RunResult.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "无法载入结果", D.friendly_error(exc))
            return
        self.state.set_result(result)
        self.toast("已载入模型，可以开始推理。", "success")


def _fill_preview_table(table: QTableWidget, frame: pd.DataFrame) -> None:
    table.clear()
    table.setColumnCount(len(frame.columns))
    table.setRowCount(len(frame))
    table.setHorizontalHeaderLabels([str(c) for c in frame.columns])
    for r in range(len(frame)):
        for c, col in enumerate(frame.columns):
            value = frame.iloc[r, c]
            if pd.isna(value):
                text = "—"
            elif isinstance(value, (float, np.floating)):
                text = D.format_number(value, 6)
            else:
                text = str(value)
            item = QTableWidgetItem(text)
            item.setFlags(Qt.ItemIsEnabled)
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(r, c, item)
