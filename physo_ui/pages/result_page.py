"""Result inspection page: expression, metrics, Pareto front and plots."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import data as D
from .. import engine as E
from .. import theme
from ..mathrender import MathLabel
from ..plotting import PlotCanvas, draw_history, draw_parity, draw_pareto, draw_residuals
from ..state import AppState
from ..units import format_units
from ..widgets import (
    Card,
    Collapsible,
    EmptyState,
    GradientCard,
    IconBadge,
    InfoRow,
    MetricTile,
    ScrollPage,
    SegmentedControl,
    badge,
    label,
    section_title,
)


class ResultPage(ScrollPage):
    predict_requested = Signal()
    train_requested = Signal()

    def __init__(self, state: AppState, toast, parent=None):
        super().__init__("结果", "查看最佳表达式、拟合指标与精度–复杂度权衡。", parent)
        self.state = state
        self.toast = toast
        self._build()
        self.state.result_changed.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ UI #
    def _build(self) -> None:
        self.empty = EmptyState(
            "还没有结果",
            "完成一次训练，或载入之前保存的 result.json。",
            "result",
        )
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        load_btn = QPushButton("载入结果…")
        load_btn.setProperty("variant", "secondary")
        load_btn.clicked.connect(self.load_result)
        btn_row.addWidget(load_btn)
        go_btn = QPushButton("前往训练  →")
        go_btn.setProperty("variant", "primary")
        go_btn.clicked.connect(self.train_requested.emit)
        btn_row.addWidget(go_btn)
        btn_row.addStretch(1)
        self.empty.add_action_layout(btn_row)
        self.body_layout.addWidget(self.empty)

        self.content = QWidget()
        lay = QVBoxLayout(self.content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        self.body_layout.addWidget(self.content)
        self._build_hero(lay)
        self._build_metrics(lay)
        details = QHBoxLayout()
        details.setSpacing(16)
        details.addWidget(self._build_expression_card(), 3)
        details.addWidget(self._build_summary_card(), 2)
        lay.addLayout(details)
        self._build_plots(lay)
        self._build_pareto(lay)
        self.body_layout.addStretch(1)

    def _build_hero(self, parent_layout) -> None:
        self.hero = GradientCard()
        lay = QVBoxLayout(self.hero)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(8)
        top = QHBoxLayout()
        top.setSpacing(8)
        icon = IconBadge("sparkles", "#FFFFFF", "rgba(255,255,255,0.18)", 36)
        top.addWidget(icon)
        texts = QVBoxLayout()
        texts.setSpacing(1)
        title = label("发现的表达式", "HeroTitle")
        self.hero_sub = label("", "HeroSub", wrap=True)
        texts.addWidget(title)
        texts.addWidget(self.hero_sub)
        top.addLayout(texts, 1)
        self.copy_btn = QPushButton("复制公式")
        self.copy_btn.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,0.16); color:white; border-radius:9px; padding:7px 14px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.28); }"
        )
        self.copy_btn.clicked.connect(self._copy_expression)
        top.addWidget(self.copy_btn, alignment=Qt.AlignTop)
        lay.addLayout(top)
        self.hero_formula = MathLabel("", color="#FFFFFF", fontsize=25, min_height=72)
        self.hero_formula.clear_formula("—")
        lay.addWidget(self.hero_formula)
        self.hero_raw = label("", "HeroSub", wrap=True, selectable=True)
        self.hero_raw.setStyleSheet("color:rgba(255,255,255,0.75); font-family:Consolas, monospace;")
        lay.addWidget(self.hero_raw)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.export_btn = QPushButton("导出结果…")
        self.export_btn.setStyleSheet(
            "QPushButton { background:white; color:#0062CC; border-radius:9px; padding:8px 14px; font-weight:600; }"
            "QPushButton:hover { background:#F2F7FF; }"
        )
        self.export_btn.clicked.connect(self.export_result)
        actions.addWidget(self.export_btn)
        self.folder_btn = QPushButton("打开运行文件夹")
        self.folder_btn.setStyleSheet(
            "QPushButton { background:rgba(255,255,255,0.16); color:white; border-radius:9px; padding:8px 14px; }"
            "QPushButton:hover { background:rgba(255,255,255,0.28); }"
        )
        self.folder_btn.clicked.connect(self._open_folder)
        actions.addWidget(self.folder_btn)
        actions.addStretch(1)
        self.predict_btn = QPushButton("用这个模型推理  →")
        self.predict_btn.setStyleSheet(
            "QPushButton { background:white; color:#0062CC; border-radius:9px; padding:8px 14px; font-weight:700; }"
            "QPushButton:hover { background:#F2F7FF; }"
        )
        self.predict_btn.clicked.connect(self.predict_requested.emit)
        actions.addWidget(self.predict_btn)
        lay.addLayout(actions)
        parent_layout.addWidget(self.hero)

    def _build_metrics(self, parent_layout) -> None:
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)
        self.r2_train_tile = MetricTile("R² · 训练集", "—", "越接近 1 越好", theme.GREEN)
        self.r2_eval_tile = MetricTile("R² · 评估集", "—", "留出集泛化", theme.BLUE)
        self.rmse_tile = MetricTile("RMSE · 训练集", "—", "原始量纲", theme.ORANGE)
        self.mae_tile = MetricTile("MAE · 训练集", "—", "平均绝对误差", theme.PURPLE)
        self.complexity_tile = MetricTile("表达式复杂度", "—", "token 复杂度之和", theme.BLUE)
        self.duration_tile = MetricTile("训练用时", "—", "epochs / 候选数", theme.GREEN)
        tiles = [
            self.r2_train_tile, self.r2_eval_tile, self.rmse_tile,
            self.mae_tile, self.complexity_tile, self.duration_tile,
        ]
        for index, tile in enumerate(tiles):
            grid.addWidget(tile, index // 3, index % 3)
        parent_layout.addLayout(grid)

    def _build_expression_card(self) -> Card:
        card = Card(padding=20)
        head = QHBoxLayout()
        head.addWidget(section_title("表达式详情", "渲染后的数学形式与可复制文本"), 1)
        self.source_badge = badge("最佳", "blue")
        head.addWidget(self.source_badge)
        card.layout.addLayout(head)
        self.formula_math = MathLabel("", fontsize=20, min_height=62)
        self.formula_math.clear_formula("—")
        card.layout.addWidget(self.formula_math)
        self.pretty_label = label("", "Caption", wrap=True, selectable=True)
        self.pretty_label.setStyleSheet(
            "font-family: Consolas, 'Cascadia Code', monospace; color:#8E8E93;"
        )
        card.layout.addWidget(self.pretty_label)
        self.const_label = label("", "Caption", wrap=True, selectable=True)
        card.layout.addWidget(self.const_label)
        self.latex_label = label("", "Caption", wrap=True, selectable=True)
        card.layout.addWidget(self.latex_label)
        self.meta_label = label("", "Caption", wrap=True, selectable=True)
        card.layout.addWidget(self.meta_label)
        card.layout.addStretch(1)
        meta = QHBoxLayout()
        meta.addStretch(1)
        copy_infix = QPushButton("复制单行公式")
        copy_infix.setProperty("variant", "link")
        copy_infix.clicked.connect(self._copy_expression)
        meta.addWidget(copy_infix)
        copy_latex = QPushButton("复制 LaTeX")
        copy_latex.setProperty("variant", "link")
        copy_latex.clicked.connect(self._copy_latex)
        meta.addWidget(copy_latex)
        card.layout.addLayout(meta)
        return card

    def _build_summary_card(self) -> Card:
        card = Card(padding=20)
        card.layout.addWidget(section_title("训练摘要", "数据、配置与输出"))
        self.summary_rows: dict[str, object] = {}
        for title in (
            "数据集", "目标变量", "输入变量", "样本 / 维度", "训练 / 评估",
            "量纲分析", "搜索预设", "批大小", "迭代轮数", "计算设备",
            "运行时间", "运行目录",
        ):
            row = InfoRow(title)
            self.summary_rows[title] = row
            card.layout.addWidget(row)
        self.units_summary = label("", "Caption", wrap=True, selectable=True)
        card.layout.addWidget(self.units_summary)
        card.layout.addStretch(1)
        return card

    def _build_plots(self, parent_layout) -> None:
        card = Card(padding=18)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(section_title("拟合诊断", "预测值 vs 真实值，以及残差分布"), 1)
        self.plot_segment = SegmentedControl(["训练集", "评估集"], current=0)
        self.plot_segment.currentChanged.connect(lambda _: self._refresh_plots())
        head.addWidget(self.plot_segment)
        card.layout.addLayout(head)
        canvases = QHBoxLayout()
        canvases.setSpacing(14)
        self.parity_canvas = PlotCanvas(4.4, 3.0)
        self.residual_canvas = PlotCanvas(4.4, 3.0)
        self.parity_canvas.clear_message("训练完成后显示")
        self.residual_canvas.clear_message("训练完成后显示")
        canvases.addWidget(self.parity_canvas)
        canvases.addWidget(self.residual_canvas)
        card.layout.addLayout(canvases)
        self.plot_note = label("", "Caption", wrap=True)
        card.layout.addWidget(self.plot_note)
        parent_layout.addWidget(card)

    def _build_pareto(self, parent_layout) -> None:
        row = QHBoxLayout()
        row.setSpacing(16)
        table_card = Card(padding=18)
        head = QHBoxLayout()
        head.addWidget(section_title("Pareto 前沿", "点击一行，切换到该表达式"), 1)
        self.candidate_count = badge("0 个候选", "gray")
        head.addWidget(self.candidate_count)
        table_card.layout.addLayout(head)
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["来源", "复杂度", "长度", "R²训练", "R²评估", "奖励", "表达式"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setMinimumHeight(250)
        self.table.setMaximumHeight(320)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 6):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.cellClicked.connect(self._on_row_clicked)
        table_card.layout.addWidget(self.table)
        self.candidate_math = MathLabel("", fontsize=18, min_height=54)
        self.candidate_math.clear_formula("选择一行查看渲染后的表达式")
        table_card.layout.addWidget(self.candidate_math)
        self.candidate_caption = label("", "Caption", wrap=True, selectable=True)
        table_card.layout.addWidget(self.candidate_caption)
        row.addWidget(table_card, 3)

        chart_card = Card(padding=18)
        chart_card.layout.addWidget(section_title("精度–复杂度", "上方/左侧越优"))
        self.pareto_canvas = PlotCanvas(4.0, 3.0)
        self.pareto_canvas.clear_message("训练完成后显示")
        chart_card.layout.addWidget(self.pareto_canvas)
        chart_card.layout.addWidget(
            label("Pareto 前沿上的每个点都是一个可选的解析表达式。", "Caption", wrap=True)
        )
        row.addWidget(chart_card, 2)
        parent_layout.addLayout(row)

        hist_card = Card(padding=18)
        hist_head = QHBoxLayout()
        hist_head.addWidget(section_title("训练历史", "最佳与平均 R² 随 epoch 的变化"), 1)
        self.history_badge = badge("—", "gray")
        hist_head.addWidget(self.history_badge)
        hist_card.layout.addLayout(hist_head)
        self.history_canvas = PlotCanvas(8.0, 2.8)
        self.history_canvas.clear_message("训练完成后显示")
        hist_card.layout.addWidget(self.history_canvas)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(
            ["Epoch", "最佳 R²", "平均 R²", "Loss", "复杂度", "本 epoch 耗时"]
        )
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setShowGrid(False)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setMaximumHeight(220)
        self.history_table.setVisible(False)
        hist_card.layout.addWidget(self.history_table)
        parent_layout.addWidget(hist_card)

    # -------------------------------------------------------------- refresh #
    def refresh(self) -> None:
        result = self.state.result
        self.empty.setVisible(result is None)
        self.content.setVisible(result is not None)
        if result is None:
            return
        active = result.active
        self.hero_sub.setText(
            f"{result.summary.get('name', '')} · 目标 {result.y_name} · "
            f"{result.epochs_done} epochs · {_duration(result.duration)} · "
            f"复杂度 {active.complexity}"
        )
        self.hero_formula.set_latex(active.latex, fallback=active.display_expression)
        self.hero_raw.setText(active.display_expression or active.expression)
        mae = float("nan")
        if result.data is not None and result.y_train is not None:
            try:
                pred = result.predict_array(result.X_train)
                mae = D.regression_metrics(result.y_train, pred)["mae"]
            except Exception:
                pass
        self.r2_train_tile.set_value(D.format_number(active.r2_train, 6), "训练集拟合优度")
        self.r2_eval_tile.set_value(
            D.format_number(active.r2_eval, 6) if _finite(active.r2_eval) else "—",
            "留出集泛化",
        )
        self.rmse_tile.set_value(
            D.format_number(active.rmse_train, 6),
            f"训练集 · n={result.summary.get('n_train', '—')}",
        )
        self.mae_tile.set_value(D.format_number(mae, 6), "平均绝对误差 · 原始量纲")
        self.complexity_tile.set_value(
            str(active.complexity),
            f"长度 {active.length} · 奖励 {D.format_number(active.reward, 4)}",
        )
        self.duration_tile.set_value(
            f"{result.duration:.1f}s",
            f"{result.epochs_done} epochs · {len(result.candidates)} 候选",
        )
        self.source_badge.setText("训练得到" if active.source == "best" else "Pareto 备选")
        self.source_badge.setObjectName("Badge" if active.source == "best" else "BadgeGray")
        _repolish(self.source_badge)
        self.formula_math.set_latex(active.latex, fallback=active.pretty or active.display_expression)
        self.pretty_label.setText(active.display_expression)
        if active.free_consts:
            consts = " · ".join(f"{k} = {D.format_number(v, 6)}" for k, v in active.free_consts.items())
            self.const_label.setText("自由常数：" + consts)
        else:
            self.const_label.setText("自由常数：无")
        self.latex_label.setText("LaTeX：" + (active.latex or "—"))
        norm_note = " · 训练时使用了标准化（上方已换算回原始量纲）" if result.normalize else ""
        self.meta_label.setText(f"模型空间表达式：{active.expression}{norm_note}")
        self._update_summary(result, active)
        self._fill_table(result)
        self.candidate_count.setText(f"{len(result.candidates)} 个候选")
        self.candidate_math.set_latex(active.latex, fallback=active.display_expression)
        self.candidate_caption.setText(
            f"{'最佳表达式' if active.source == 'best' else 'Pareto 备选'} · "
            f"R² {D.format_number(active.r2_train, 6)} · 复杂度 {active.complexity} · "
            f"长度 {active.length}"
        )
        draw_pareto(self.pareto_canvas, result.candidates)
        if result.history:
            draw_history(self.history_canvas, result.history)
            self.history_badge.setText(f"{len(result.history)} epochs")
            self._fill_history(result.history)
        else:
            self.history_canvas.clear_message("没有保存训练历史")
            self.history_table.setVisible(False)
        self._refresh_plots()
        self.folder_btn.setEnabled(bool(result.run_dir))
        self.export_btn.setEnabled(True)

    def _update_summary(self, result: E.RunResult, active: E.Candidate) -> None:
        cfg = result.config or {}
        summary = result.summary
        values = {
            "数据集": summary.get("name", "—"),
            "目标变量": result.y_name if result.y_name == result.y_column else f"{result.y_name} ({result.y_column})",
            "输入变量": ", ".join(result.x_names),
            "样本 / 维度": f"{summary.get('n_samples', '—')} / {summary.get('n_dim', '—')}",
            "训练 / 评估": f"{summary.get('n_train', '—')} / {summary.get('n_eval', 0)}",
            "量纲分析": "开启" if result.use_units else "关闭",
            "搜索预设": str(cfg.get("preset", summary.get("preset", "—"))),
            "批大小": f"{cfg.get('batch_size', summary.get('batch_size', '—')):,}" if isinstance(cfg.get("batch_size", summary.get("batch_size")), int) else str(cfg.get("batch_size", "—")),
            "迭代轮数": f"{result.epochs_done} / {cfg.get('epochs', summary.get('epochs', '—'))}",
            "计算设备": str(cfg.get("device", "—")),
            "运行时间": f"{result.duration:.1f} s",
            "运行目录": Path(result.run_dir).name if result.run_dir else "—",
        }
        for title, row in self.summary_rows.items():
            row.set_value(str(values.get(title, "—")))
        unit_parts = [
            f"{name} [{format_units(vec)}]" for name, vec in zip(result.x_names, result.x_units)
        ]
        unit_parts.append(f"{result.y_name} [{format_units(result.y_units)}]")
        self.units_summary.setText("变量量纲：" + " · ".join(unit_parts))

    def _fill_history(self, history: list[dict]) -> None:
        rows = history[-500:]
        self.history_table.setVisible(bool(rows))
        self.history_table.setRowCount(len(rows))
        for r, entry in enumerate(rows):
            values = [
                str(int(entry.get("epoch", r)) + 1),
                D.format_number(entry.get("best_r2"), 6),
                D.format_number(entry.get("mean_r2"), 5),
                D.format_number(entry.get("loss"), 5),
                str(entry.get("best_complexity", "—")),
                f"{D.format_number(entry.get('epoch_time', 0), 3)} s",
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if c:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.history_table.setItem(r, c, item)

    def _fill_table(self, result: E.RunResult) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(result.candidates))
        for i, cand in enumerate(result.candidates):
            values = [
                "最佳" if cand.source == "best" else "备选",
                str(cand.complexity),
                str(cand.length),
                D.format_number(cand.r2_train, 4),
                D.format_number(cand.r2_eval, 4) if _finite(cand.r2_eval) else "—",
                D.format_number(cand.reward, 4),
                cand.display_expression,
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if 1 <= c <= 5:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, c, item)
        self.table.blockSignals(False)
        self.table.selectRow(max(0, min(result.active_index, len(result.candidates) - 1)))

    def _refresh_plots(self) -> None:
        result = self.state.result
        if result is None:
            return
        if result.data is None:
            self.parity_canvas.clear_message("载入的结果不含原始数据，请查看运行文件夹中的 PNG")
            self.residual_canvas.clear_message("载入的结果不含原始数据")
            self.plot_note.setText(
                "提示：训练时自动保存的 fit_train.png / fit_eval.png / residuals_*.png 位于运行目录。"
            )
            return
        use_eval = self.plot_segment.current_index() == 1
        if use_eval:
            if result.y_eval is None or len(result.y_eval) == 0:
                self.parity_canvas.clear_message("本次训练没有留出评估集")
                self.residual_canvas.clear_message("本次训练没有留出评估集")
                self.plot_note.setText("如需评估集，请在数据页开启「留出评估集」。")
                return
            y_true = result.y_eval
            y_pred = result.predict_array(result.X_eval)
            title = "评估集：预测 vs 真实"
        else:
            y_true = result.y_train
            y_pred = result.predict_array(result.X_train)
            title = "训练集：预测 vs 真实"
        draw_parity(self.parity_canvas, y_true, y_pred, title)
        draw_residuals(self.residual_canvas, y_true, y_pred)
        metrics = D.regression_metrics(y_true, y_pred)
        self.plot_note.setText(
            f"n = {metrics['n']} · R² = {D.format_number(metrics['r2'], 6)} · "
            f"RMSE = {D.format_number(metrics['rmse'], 6)} · MAE = {D.format_number(metrics['mae'], 6)}"
        )

    def _on_row_clicked(self, row: int, _col: int) -> None:
        result = self.state.result
        if result is None:
            return
        if 0 <= row < len(result.candidates):
            result.activate(row)
            self.refresh()
            self.toast("已切换推理公式。", "info")

    # --------------------------------------------------------------- actions #
    def _copy_expression(self) -> None:
        result = self.state.result
        if result is None:
            return
        QApplication.clipboard().setText(result.active.display_expression)
        self.toast("公式已复制到剪贴板。", "success")

    def _copy_latex(self) -> None:
        result = self.state.result
        if result is None:
            return
        QApplication.clipboard().setText(result.active.latex)
        self.toast("LaTeX 已复制。", "success")

    def _open_folder(self) -> None:
        result = self.state.result
        if result is None or not result.run_dir:
            self.toast("没有运行目录。", "warn")
            return
        path = Path(result.run_dir)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self.toast("运行目录不存在。", "warn")

    def load_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "载入结果",
            str(E.RUNS_DIR),
            "结果文件 (result.json *.json);;所有文件 (*.*)",
        )
        if not path:
            return
        try:
            result = E.RunResult.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "无法载入结果", D.friendly_error(exc))
            return
        self.state.set_result(result)
        self.toast("已载入结果，可直接用于推理。", "success")

    def export_result(self) -> None:
        result = self.state.result
        if result is None:
            self.toast("没有可导出的结果。", "warn")
            return
        target = QFileDialog.getExistingDirectory(self, "选择导出文件夹", str(E.RUNS_DIR))
        if not target:
            return
        try:
            path = result.save_to_dir(Path(target) / "export")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", D.friendly_error(exc))
            return
        self.toast(f"结果已导出到 {path}", "success")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _finite(value: float) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _duration(seconds: float) -> str:
    seconds = int(max(0, seconds or 0))
    m, s = divmod(seconds, 60)
    return f"{m} 分 {s} 秒" if m else f"{s} 秒"


def _repolish(widget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
