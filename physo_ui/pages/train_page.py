"""Training page with live formula, metrics, charts and full epoch history."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import data as D
from .. import engine as E
from ..mathrender import MathLabel
from ..plotting import PlotCanvas, draw_history
from ..state import AppState
from ..widgets import (
    Card,
    EmptyState,
    InfoRow,
    GradientCard,
    MetricTile,
    ScrollPage,
    SegmentedControl,
    badge,
    label,
    section_title,
)


class TrainPage(ScrollPage):
    result_requested = Signal()
    data_requested = Signal()

    def __init__(self, state: AppState, toast, parent=None):
        super().__init__("训练", "在后台运行 PhySO，实时查看当前最佳公式与完整训练过程。", parent)
        self.state = state
        self.toast = toast
        self._worker: E.TrainingWorker | None = None
        self._started_at = 0.0
        self._last_payload: dict = {}
        self._ticker = QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._tick_elapsed)
        self._build()
        self.state.prepared_changed.connect(self._refresh_summary)
        self.state.training_changed.connect(self._on_training_changed)
        self._refresh_summary()

    # ------------------------------------------------------------------ UI #
    def _build(self) -> None:
        view_row = QHBoxLayout()
        view_row.setSpacing(10)
        self.view_switch = SegmentedControl(["训练概览", "过程与配置"], current=0)
        self.view_switch.currentChanged.connect(self._on_view_mode)
        view_row.addWidget(self.view_switch)
        view_row.addStretch(1)
        self.result_btn_top = QPushButton("查看结果  →")
        self.result_btn_top.setProperty("variant", "secondary")
        self.result_btn_top.setEnabled(False)
        self.result_btn_top.clicked.connect(self.result_requested.emit)
        view_row.addWidget(self.result_btn_top)
        self.body_layout.addLayout(view_row)

        self.view_stack = QStackedWidget()
        overview = QWidget()
        overview_lay = QVBoxLayout(overview)
        overview_lay.setContentsMargins(0, 0, 0, 0)
        overview_lay.setSpacing(14)
        details = QWidget()
        details_lay = QVBoxLayout(details)
        details_lay.setContentsMargins(0, 0, 0, 0)
        details_lay.setSpacing(14)
        self.view_stack.addWidget(overview)
        self.view_stack.addWidget(details)
        self.body_layout.addWidget(self.view_stack, 1)

        self.hero = GradientCard()
        self.hero.setMinimumHeight(112)
        self.hero.setMaximumHeight(122)
        hero_lay = QHBoxLayout(self.hero)
        hero_lay.setContentsMargins(24, 20, 24, 20)
        hero_lay.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(4)
        self.hero_title = label("准备就绪", "HeroTitle")
        self.hero_sub = label("检查数据与参数，然后开始符号回归搜索。", "HeroSub", wrap=True)
        left.addWidget(self.hero_title)
        left.addWidget(self.hero_sub)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.setStyleSheet(
            "QProgressBar { background: rgba(255,255,255,0.28); border-radius:4px; }"
            "QProgressBar::chunk { background: white; border-radius:4px; }"
        )
        left.addSpacing(8)
        left.addWidget(self.progress)
        hero_lay.addLayout(left, 1)
        buttons = QVBoxLayout()
        buttons.setSpacing(8)
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setMinimumWidth(132)
        self.start_btn.setStyleSheet(
            "QPushButton { background:white; color:#0062CC; border-radius:11px;"
            "font-size:14px; font-weight:700; }"
            "QPushButton:hover { background:#F2F7FF; }"
            "QPushButton:disabled { background:rgba(255,255,255,.55); color:#8AAFD6; }"
        )
        self.start_btn.clicked.connect(self.start_training)
        buttons.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setStyleSheet(
            "QPushButton { background:rgba(255,255,255,0.16); color:white; border-radius:10px; }"
            "QPushButton:hover { background:rgba(255,255,255,0.26); }"
            "QPushButton:disabled { color:rgba(255,255,255,0.45); }"
        )
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self.stop_training)
        buttons.addWidget(self.stop_btn)
        hero_lay.addLayout(buttons)
        overview_lay.addWidget(self.hero)

        stats = QHBoxLayout()
        stats.setSpacing(14)
        self.epoch_tile = MetricTile("当前轮次", "0 / 0", "等待开始", "#007AFF")
        self.r2_tile = MetricTile("最佳 R²", "—", "训练集拟合优度", "#34C759")
        self.reward_tile = MetricTile("最佳奖励", "—", "SquashedNRMSE ∈ [0,1]", "#FF9500")
        self.time_tile = MetricTile("已用时间", "00:00", "—", "#5856D6")
        for tile in (self.epoch_tile, self.r2_tile, self.reward_tile, self.time_tile):
            tile.setMaximumHeight(132)
            stats.addWidget(tile)
        overview_lay.addLayout(stats)

        self.empty = EmptyState(
            "还没有训练数据",
            "先导入数据并选择变量，然后回到这里开始训练。",
            "train",
        )
        self.empty.setVisible(False)
        self.empty.add_action(self._empty_button())

        self.formula_card = Card(padding=14)
        self.formula_card.layout.setSpacing(5)
        self.formula_card.setMaximumHeight(132)
        head = QHBoxLayout()
        head.addWidget(section_title("当前最佳表达式", "每个 epoch 更新一次"), 1)
        self.best_badge = badge("等待第一个 epoch", "gray")
        head.addWidget(self.best_badge)
        self.copy_btn = QPushButton("复制")
        self.copy_btn.setProperty("variant", "link")
        self.copy_btn.clicked.connect(self._copy_expression)
        head.addWidget(self.copy_btn)
        self.formula_card.layout.addLayout(head)
        self.formula_label = MathLabel("", fontsize=20, min_height=42)
        self.formula_label.setMaximumHeight(46)
        self.formula_label.clear_formula("训练开始后，这里会渲染当前最优公式")
        self.formula_card.layout.addWidget(self.formula_label)
        self.raw_formula = label("—", "Caption", wrap=True, selectable=True)
        self.raw_formula.setStyleSheet(
            "font-family: Consolas, 'Cascadia Code', monospace; color:#8E8E93;"
        )
        self.raw_formula.setMaximumHeight(20)
        self.formula_card.layout.addWidget(self.raw_formula)
        self.formula_note = label("", "Caption", wrap=True)
        self.formula_note.setVisible(False)
        overview_lay.addWidget(self.formula_card)

        plot_card = Card(padding=18)
        self.plot_card = plot_card
        head = QHBoxLayout()
        head.addWidget(section_title("训练曲线", "最佳 R² 与每轮平均 R²"), 1)
        self.best_chart_badge = badge("等待开始", "gray")
        head.addWidget(self.best_chart_badge)
        plot_card.layout.addLayout(head)
        self.chart = PlotCanvas(5.6, 3.0)
        self.chart.setMinimumHeight(220)
        self.chart.setMaximumHeight(180)
        self.chart.clear_message("训练开始后显示实时曲线")
        plot_card.layout.addWidget(self.chart)
        plot_card.setMaximumHeight(250)
        overview_lay.addWidget(plot_card)
        overview_lay.addStretch(1)

        self.status_card = Card(padding=18)
        self.status_card.layout.addWidget(section_title("运行状态", "当前 batch 与速度信息"))
        self.status_rows: dict[str, InfoRow] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(2)
        for i, title in enumerate(
            (
                "本 epoch 耗时", "预计剩余", "最佳奖励", "平均 R²",
                "Loss", "表达式复杂度", "表达式长度", "物理候选",
                "有效候选", "设备",
            )
        ):
            row = InfoRow(title)
            self.status_rows[title] = row
            grid.addWidget(row, i % 5, i // 5)
        self.status_card.layout.addLayout(grid)
        self.status_card.layout.addStretch(1)
        self.status_card.setMaximumHeight(320)
        details_lay.addWidget(self.status_card)

        self.process_card = Card(padding=18)
        head = QHBoxLayout()
        head.addWidget(section_title("训练过程", "逐轮指标与运行日志"), 1)
        self.history_badge = badge("0 轮", "gray")
        head.addWidget(self.history_badge)
        self.process_segment = SegmentedControl(["逐轮指标", "运行日志"], current=0)
        self.process_segment.currentChanged.connect(self._on_process_mode)
        head.addWidget(self.process_segment)
        self.process_card.layout.addLayout(head)
        self.process_stack = QStackedWidget()

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(9)
        self.history_table.setHorizontalHeaderLabels(
            ["Epoch", "最佳 R²", "平均 R²", "奖励", "Loss", "复杂度", "长度", "物理候选", "耗时"]
        )
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setShowGrid(False)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setMinimumHeight(250)
        self.history_table.setMaximumHeight(310)
        self.process_stack.addWidget(self.history_table)

        self.console = QPlainTextEdit()
        self.console.setObjectName("Console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        self.console.setMinimumHeight(250)
        self.console.setStyleSheet(
            "background:#141417; color:#E7E7EC; border:none; border-radius:14px; padding:12px;"
            "font-family: Consolas, 'Cascadia Code', monospace; font-size:11px;"
        )
        self.process_stack.addWidget(self.console)
        self.process_card.layout.addWidget(self.process_stack)
        details_lay.addWidget(self.process_card)

        self.summary_card = Card(padding=20)
        head = QHBoxLayout()
        head.addWidget(section_title("本次训练", "数据、配置与输出"), 1)
        self.folder_btn = QPushButton("打开运行文件夹")
        self.folder_btn.setProperty("variant", "secondary")
        self.folder_btn.clicked.connect(self._open_folder)
        self.folder_btn.setEnabled(False)
        head.addWidget(self.folder_btn)
        self.result_btn = QPushButton("查看结果  →")
        self.result_btn.setProperty("variant", "primary")
        self.result_btn.setEnabled(False)
        self.result_btn.clicked.connect(self.result_requested.emit)
        head.addWidget(self.result_btn)
        self.summary_card.layout.addLayout(head)

        self.summary_grid = QHBoxLayout()
        self.summary_grid.setSpacing(24)
        self.summary_left = QVBoxLayout()
        self.summary_right = QVBoxLayout()
        self.summary_left.setSpacing(2)
        self.summary_right.setSpacing(2)
        self.summary_grid.addLayout(self.summary_left, 1)
        self.summary_grid.addLayout(self.summary_right, 1)
        self.summary_card.layout.addLayout(self.summary_grid)
        details_lay.addWidget(self.summary_card)
        details_lay.addStretch(1)
        self.body_layout.addWidget(self.empty)

        self._summary_rows: list[InfoRow] = []
        self._build_summary_rows()

    def _empty_button(self) -> QPushButton:
        btn = QPushButton("前往数据  →")
        btn.setProperty("variant", "primary")
        btn.clicked.connect(self.data_requested.emit)
        return btn

    def _build_summary_rows(self) -> None:
        titles = [
            "数据集", "样本 / 维度", "目标变量", "训练 / 评估",
            "搜索预设", "批大小", "表达式长度", "计算设备",
            "迭代轮数", "学习率", "自由常数", "固定常数",
            "随机种子", "CPU 线程", "常数优化步数", "算子数量 / 量纲",
        ]
        for i, title in enumerate(titles):
            row = InfoRow(title)
            if i % 2 == 0:
                self.summary_left.addWidget(row)
            else:
                self.summary_right.addWidget(row)
            self._summary_rows.append(row)

    # -------------------------------------------------------------- summary #
    def _refresh_summary(self) -> None:
        prepared = self.state.prepared
        self.empty.setVisible(prepared is None)
        self.view_switch.setVisible(prepared is not None)
        self.view_stack.setVisible(prepared is not None)
        for widget in (self.formula_card, self.process_card, self.status_card, self.summary_card, self.plot_card):
            widget.setVisible(prepared is not None)
        self.start_btn.setEnabled(prepared is not None and not self._is_running())
        if prepared is None:
            self.hero_title.setText("准备就绪")
            self.hero_sub.setText("请先在「数据」页导入并选择数据。")
            return
        cfg = self.state.config
        self.formula_note.setText(
            "训练在标准化空间进行；结果页会换算回原始量纲。"
            if prepared.normalize
            else "训练使用原始变量；公式与量纲直接对应。"
        )
        values = [
            prepared.name,
            f"{prepared.n_samples} / {prepared.n_dim}",
            f"{prepared.y_name} ({prepared.y_col})",
            f"{prepared.n_train} / {prepared.n_eval}" if prepared.n_eval else f"{prepared.n_train} / —",
            f"{E.PRESETS.get(cfg.preset, {}).get('name', cfg.preset)} ({cfg.preset})",
            f"{cfg.batch_size:,} / epoch",
            f"{cfg.max_length} tokens",
            {"auto": "自动", "cuda": "CUDA", "cpu": "CPU"}.get(cfg.device, cfg.device),
            f"{cfg.epochs:,}",
            f"{cfg.lr:g}",
            f"{cfg.n_free_consts} 个",
            ", ".join(f"{v:g}" for v in cfg.fixed_consts) or "无",
            str(cfg.seed),
            str(cfg.cpu_threads or "自动"),
            str(cfg.const_opti_steps),
            f"{len(cfg.ops)} / "
            + ("量纲分析" if prepared.use_units and prepared.has_units else ("量纲已设" if prepared.has_units else "无量纲")),
        ]
        for row, value in zip(self._summary_rows, values):
            row.set_value(str(value))
        if not self._is_running() and self.state.result is None:
            self.epoch_tile.set_value(f"0 / {cfg.epochs:,}", "等待开始")
            self.hero_title.setText("准备就绪")
            self.hero_sub.setText(
                f"{prepared.name} · {prepared.describe()} · 点击「开始训练」启动搜索。"
            )
        if not self._is_running() and self.state.result is not None:
            self.hero_title.setText("训练已完成")
            self.hero_sub.setText(
                f"最佳 R² = {D.format_number(self.state.result.active.r2_train, 6)} · "
                "可查看结果，或直接开始新的训练。"
            )

    def _is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # -------------------------------------------------------------- training #
    def start_training(self) -> None:
        if self._is_running():
            return
        prepared = self.state.prepared
        if prepared is None:
            self.toast("请先在「数据」页准备训练数据。", "error")
            return
        cfg = self.state.config
        run_dir = E.make_run_dir(prepared.name)
        self._worker = E.TrainingWorker(prepared, cfg, run_dir, self)
        self.state.set_worker(self._worker)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log)
        self._worker.succeeded.connect(self._on_succeeded)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_thread_finished)

        self.console.clear()
        self.chart.clear_message("正在初始化…")
        self.history_table.setRowCount(0)
        self.history_badge.setText("0 轮")
        self.formula_label.clear_formula("训练开始后，这里会渲染当前最优公式")
        self.raw_formula.setText("—")
        self.hero_title.setText("正在初始化…")
        self.hero_sub.setText("加载 PyTorch 与 PhySO，请稍候。")
        self.progress.setValue(0)
        self.epoch_tile.set_value(f"0 / {cfg.epochs}", "初始化引擎")
        self.r2_tile.set_value("—", "训练集拟合优度")
        self.reward_tile.set_value("—", "SquashedNRMSE ∈ [0,1]")
        self.time_tile.set_value("00:00", "运行中")
        self.best_badge.setText("初始化")
        self.best_chart_badge.setText("初始化")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setVisible(True)
        self.result_btn.setEnabled(False)
        self.result_btn_top.setEnabled(False)
        self.folder_btn.setEnabled(True)
        self._started_at = time.time()
        self._ticker.start()
        self._append_log(f"运行目录：{run_dir}")
        self._append_log(f"启动：{cfg.preset} · {cfg.epochs:,} epochs · batch {cfg.batch_size:,}")
        self._worker.start()

    def stop_training(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.hero_title.setText("正在停止…")
            self.hero_sub.setText("当前 epoch 结束后停止，并保留已找到的最优表达式。")
            self._append_log("已发送停止请求。")

    def _on_progress(self, payload: dict) -> None:
        self._last_payload = payload
        epoch = int(payload.get("epoch", 0))
        epochs = max(1, int(payload.get("epochs", 1)))
        self.progress.setValue(int(round((epoch + 1) / epochs * 100)))
        self.epoch_tile.set_value(
            f"{epoch + 1} / {epochs}",
            f"本 epoch {D.format_number(payload.get('epoch_time', 0), 3)}s",
        )
        self.r2_tile.set_value(D.format_number(payload.get("best_r2"), 6), "训练集拟合优度")
        self.reward_tile.set_value(
            D.format_number(payload.get("best_reward"), 5), "SquashedNRMSE ∈ [0,1]"
        )
        self.time_tile.set_value(
            _fmt_duration(payload.get("elapsed", 0)),
            f"预计剩余 {_fmt_duration(payload.get('eta', 0))}",
        )
        self.hero_title.setText("正在搜索表达式")
        self.hero_sub.setText(
            f"epoch {epoch + 1} / {epochs} · 最佳 R² {D.format_number(payload.get('best_r2'), 6)} · "
            f"用时 {_fmt_duration(payload.get('elapsed', 0))} · 预计剩余 {_fmt_duration(payload.get('eta', 0))}"
        )
        latex = payload.get("best_latex") or ""
        raw = payload.get("expression") or ""
        if latex:
            self.formula_label.set_latex(latex, fallback=raw)
        elif raw:
            self.formula_label.clear_formula(raw)
        self.raw_formula.setText(raw or "—")
        self.best_badge.setText(f"最佳 R² {D.format_number(payload.get('best_r2'), 6)}")
        self.best_badge.setObjectName("BadgeGreen")
        _repolish(self.best_badge)
        self.best_chart_badge.setText(f"epoch {epoch + 1}")
        self.best_chart_badge.setObjectName("Badge")
        _repolish(self.best_chart_badge)
        self._update_status(payload)
        self._append_history(payload)
        draw_history(self.chart, payload.get("history", []))
        self.result_btn.setEnabled(False)

    def _update_status(self, payload: dict) -> None:
        values = {
            "本 epoch 耗时": f"{D.format_number(payload.get('epoch_time', 0), 4)} s",
            "预计剩余": _fmt_duration(payload.get("eta", 0)),
            "最佳奖励": D.format_number(payload.get("best_reward"), 5),
            "平均 R²": D.format_number(payload.get("mean_r2"), 5),
            "Loss": D.format_number(payload.get("loss"), 5),
            "表达式复杂度": str(payload.get("best_complexity", "—")),
            "表达式长度": str(payload.get("best_length", "—")),
            "物理候选": str(payload.get("n_physical", "—")),
            "有效候选": str(payload.get("n_rewarded", "—")),
            "设备": {"auto": "自动", "cuda": "CUDA", "cpu": "CPU"}.get(
                self.state.config.device, self.state.config.device
            ),
        }
        for title, value in values.items():
            if title in self.status_rows:
                self.status_rows[title].set_value(value)

    def _append_history(self, payload: dict) -> None:
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        values = [
            str(int(payload.get("epoch", 0)) + 1),
            D.format_number(payload.get("best_r2"), 6),
            D.format_number(payload.get("mean_r2"), 5),
            D.format_number(payload.get("best_reward"), 5),
            D.format_number(payload.get("loss"), 5),
            str(payload.get("best_complexity", "—")),
            str(payload.get("best_length", "—")),
            str(payload.get("n_physical", "—")),
            f"{D.format_number(payload.get('epoch_time', 0), 3)} s",
        ]
        for c, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if c:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.history_table.setItem(row, c, item)
        self.history_badge.setText(f"{row} 轮")
        self.history_table.scrollToBottom()

    def _on_log(self, line: str) -> None:
        self._append_log(line)

    def _append_log(self, text: str) -> None:
        self.console.appendPlainText(text)
        bar = self.console.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_succeeded(self, result: E.RunResult) -> None:
        self.state.set_result(result)
        self._ticker.stop()
        self.progress.setValue(100 if not result.stopped else self.progress.value())
        title = "已停止" if result.stopped else "训练完成"
        self.hero_title.setText(title)
        self.hero_sub.setText(
            f"找到 {len(result.candidates)} 个候选 · 最佳 R² "
            f"{D.format_number(result.active.r2_train, 6)} · 用时 {_fmt_duration(result.duration)}"
        )
        self.epoch_tile.set_value(f"{result.epochs_done} / {result.epochs_done}", "已结束")
        self.r2_tile.set_value(D.format_number(result.active.r2_train, 6), "训练集拟合优度")
        self.reward_tile.set_value(
            D.format_number(result.active.reward, 5), "SquashedNRMSE ∈ [0,1]"
        )
        self.time_tile.set_value(_fmt_duration(result.duration), "训练结束")
        self.formula_label.set_latex(
            result.active.latex, fallback=result.active.display_expression
        )
        self.raw_formula.setText(result.active.display_expression)
        self.best_badge.setText(f"最佳 R² {D.format_number(result.active.r2_train, 6)}")
        self.best_badge.setObjectName("BadgeGreen")
        _repolish(self.best_badge)
        self.result_btn.setEnabled(True)
        self.result_btn_top.setEnabled(True)
        self.folder_btn.setEnabled(bool(result.run_dir))
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self._append_log(
            f"完成：R²={D.format_number(result.active.r2_train, 6)} · "
            f"RMSE={D.format_number(result.active.rmse_train, 6)} · "
            f"复杂度={result.active.complexity}"
        )
        self.toast("训练完成，结果已自动保存。", "success")

    def _on_failed(self, message: str) -> None:
        self._ticker.stop()
        self.hero_title.setText("训练失败")
        self.hero_sub.setText(message.splitlines()[0][:160])
        self.best_badge.setText("失败")
        self.best_badge.setObjectName("BadgeOrange")
        _repolish(self.best_badge)
        self.start_btn.setEnabled(self.state.prepared is not None)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.time_tile.set_value("—", "已停止")
        self._append_log(message)
        self.toast("训练失败，请查看运行日志。", "error")

    def _on_thread_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self.state.set_worker(None)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.start_btn.setEnabled(self.state.prepared is not None)

    def _on_training_changed(self) -> None:
        running = self.state.training_active
        self.start_btn.setEnabled(not running and self.state.prepared is not None)
        self.stop_btn.setEnabled(running)

    def _tick_elapsed(self) -> None:
        if self._started_at:
            elapsed = time.time() - self._started_at
            self.time_tile.set_value(
                _fmt_duration(elapsed),
                f"预计剩余 {_fmt_duration(self._last_payload.get('eta', 0))}",
            )

    def _on_process_mode(self, index: int) -> None:
        self.process_stack.setCurrentIndex(index)

    def _on_view_mode(self, index: int) -> None:
        self.view_stack.setCurrentIndex(index)

    def _copy_expression(self) -> None:
        text = self._last_payload.get("expression") or (
            self.state.result.active.display_expression if self.state.result else ""
        )
        if text:
            QApplication.clipboard().setText(text)
            self.toast("表达式已复制。", "success")

    def _open_folder(self) -> None:
        path = ""
        if self._worker is not None:
            path = str(self._worker.run_dir)
        elif self.state.result is not None:
            path = self.state.result.run_dir
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            self.toast("还没有可打开的运行目录。", "warn")


def _fmt_duration(seconds: float | None) -> str:
    if not seconds or seconds < 0:
        return "00:00"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _repolish(widget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
