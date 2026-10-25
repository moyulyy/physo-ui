"""Model and hyper-parameter configuration page."""

from __future__ import annotations

import math
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import engine as E
from ..state import AppState
from ..units import format_units
from ..widgets import (
    Card,
    Collapsible,
    FlowLayout,
    IconBadge,
    ScrollPage,
    SegmentedControl,
    ToggleRow,
    badge,
    label,
    section_title,
)


OP_GROUPS = [
    ("基础运算", ["add", "sub", "mul", "div", "neg", "inv"]),
    ("幂与根", ["n2", "n3", "n4", "sqrt", "cbrt", "pow"]),
    ("指数与对数", ["exp", "expneg", "log", "logabs"]),
    ("三角与双曲", ["sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh"]),
    ("其他", ["abs", "erf"]),
]

OP_HINTS = {
    "add": "加法", "sub": "减法", "mul": "乘法", "div": "除法", "neg": "取负", "inv": "倒数",
    "n2": "平方", "n3": "三次方", "n4": "四次方", "sqrt": "平方根", "cbrt": "立方根", "pow": "幂",
    "exp": "指数", "expneg": "负指数", "log": "自然对数", "logabs": "绝对值的对数",
    "sin": "正弦", "cos": "余弦", "tan": "正切", "arcsin": "反正弦", "arccos": "反余弦",
    "arctan": "反正切", "sinh": "双曲正弦", "cosh": "双曲余弦", "tanh": "双曲正切",
    "abs": "绝对值", "erf": "误差函数",
}


class ParamTile(QFrame):
    """Compact iOS-settings-like parameter card."""

    def __init__(self, title: str, subtitle: str, control: QWidget, parent=None):
        super().__init__(parent)
        self.setObjectName("InnerCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(5)
        title_label = label(title, "CardTitle")
        title_label.setStyleSheet("font-size:13px;")
        lay.addWidget(title_label)
        lay.addWidget(label(subtitle, "Caption", wrap=True))
        lay.addSpacing(2)
        lay.addWidget(control)
        self.setMinimumHeight(112)


class ModelPage(ScrollPage):
    start_requested = Signal()
    edit_data_requested = Signal()

    def __init__(self, state: AppState, toast, parent=None):
        super().__init__("模型", "设置符号搜索空间、表达式规模与训练预算。", parent)
        self.state = state
        self.toast = toast
        self._syncing = False
        self._emitting = False
        self._chips: dict[str, QPushButton] = {}
        self._build()
        self.state.prepared_changed.connect(self._refresh_data)
        self.state.config_changed.connect(self._on_config_external)
        self._refresh_data()
        self._sync_from_config()

    # ------------------------------------------------------------------ UI #
    def _build(self) -> None:
        top = QHBoxLayout()
        top.setSpacing(16)
        self.data_card = self._build_data_card()
        top.addWidget(self.data_card, 3)
        self.device_card = self._build_device_card()
        top.addWidget(self.device_card, 2)
        self.body_layout.addLayout(top)

        nav = QHBoxLayout()
        nav.setSpacing(10)
        self.section_switch = SegmentedControl(
            ["搜索空间", "训练参数", "高级设置"], current=0
        )
        self.section_switch.currentChanged.connect(self._on_section_changed)
        nav.addWidget(self.section_switch)
        nav.addStretch(1)
        back = QPushButton("← 返回数据")
        back.setProperty("variant", "secondary")
        back.clicked.connect(self.edit_data_requested.emit)
        nav.addWidget(back)
        self.summary_label = label("", "Caption")
        nav.addWidget(self.summary_label)
        self.start_btn = QPushButton("保存并前往训练  →")
        self.start_btn.setProperty("variant", "primary")
        self.start_btn.setMinimumHeight(38)
        self.start_btn.clicked.connect(self._on_start)
        nav.addWidget(self.start_btn)
        self.body_layout.addLayout(nav)

        self.section_stack = QStackedWidget()
        search_page = QWidget()
        search_lay = QVBoxLayout(search_page)
        search_lay.setContentsMargins(0, 0, 0, 0)
        search_lay.setSpacing(14)
        search_lay.addWidget(self._build_preset_card())
        search_lay.addWidget(self._build_ops_card())
        search_lay.addStretch(1)
        self.section_stack.addWidget(search_page)

        params_page = QWidget()
        params_lay = QVBoxLayout(params_page)
        params_lay.setContentsMargins(0, 0, 0, 0)
        params_lay.setSpacing(14)
        params_lay.addWidget(self._build_training_card())
        params_lay.addStretch(1)
        self.section_stack.addWidget(params_page)

        advanced_page = QWidget()
        advanced_lay = QVBoxLayout(advanced_page)
        advanced_lay.setContentsMargins(0, 0, 0, 0)
        advanced_lay.setSpacing(14)
        advanced_lay.addWidget(self._build_advanced_card())
        advanced_lay.addStretch(1)
        self.section_stack.addWidget(advanced_page)
        self.body_layout.addWidget(self.section_stack)
        self.body_layout.addStretch(1)

    def _build_data_card(self) -> Card:
        card = Card(padding=18)
        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(IconBadge("data", "#007AFF", "#E8F1FF", 36))
        head.addWidget(section_title("当前数据", "训练使用、变量量纲与划分"), 1)
        edit = QPushButton("修改")
        edit.setProperty("variant", "link")
        edit.clicked.connect(self.edit_data_requested.emit)
        head.addWidget(edit)
        card.layout.addLayout(head)
        self.data_name = label("尚未准备数据", "CardTitle")
        self.data_name.setStyleSheet("font-size:16px;")
        self.data_desc = label("请先到「数据」页导入数据、选择特征并设置量纲。", "Caption", wrap=True)
        card.layout.addWidget(self.data_name)
        card.layout.addWidget(self.data_desc)
        self.data_units = label("", "Caption", wrap=True, selectable=True)
        card.layout.addWidget(self.data_units)
        self.data_badges = QHBoxLayout()
        self.data_badges.setSpacing(8)
        self.data_badges.addStretch(1)
        card.layout.addLayout(self.data_badges)
        return card

    def _build_device_card(self) -> Card:
        card = Card(padding=18)
        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(IconBadge("model", "#34C759", "#E9F9EE", 36))
        head.addWidget(section_title("运行环境", "PyTorch / CUDA / CPU"), 1)
        card.layout.addLayout(head)
        self.env_label = label("正在检测…", "CardTitle")
        self.env_detail = label("", "Caption", wrap=True)
        card.layout.addWidget(self.env_label)
        card.layout.addWidget(self.env_detail)
        self.cpu_note = label(
            "PhySO 的主要瓶颈通常在 CPU 侧的采样与常数优化；GPU 利用率偏低是正常现象。"
            " 可在下方设置 CPU 线程数。",
            "Caption",
            wrap=True,
        )
        card.layout.addWidget(self.cpu_note)
        card.layout.addStretch(1)
        return card

    def _build_preset_card(self) -> Card:
        card = Card(padding=18)
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(section_title("搜索预设", "PhySO 官方配置"), 1)
        self.preset_control = SegmentedControl(
            [E.PRESETS["config0"]["name"], E.PRESETS["config2"]["name"], E.PRESETS["config1"]["name"]],
            current=0,
        )
        self.preset_control.currentChanged.connect(self._on_preset)
        row.addWidget(self.preset_control)
        card.layout.addLayout(row)
        self.preset_note = label(E.PRESETS["config0"]["note"], "Caption", wrap=True)
        card.layout.addWidget(self.preset_note)
        return card

    def _build_ops_card(self) -> Card:
        card = Card(padding=18)
        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(section_title("符号算子", "按类别勾选搜索空间"), 1)
        recommended = QPushButton("推荐")
        recommended.setProperty("variant", "link")
        recommended.clicked.connect(self._select_recommended)
        all_btn = QPushButton("全选")
        all_btn.setProperty("variant", "link")
        all_btn.clicked.connect(lambda: self._set_ops(E.ALL_OPS))
        none_btn = QPushButton("清空")
        none_btn.setProperty("variant", "link")
        none_btn.clicked.connect(lambda: self._set_ops([]))
        for btn in (recommended, all_btn, none_btn):
            head.addWidget(btn)
        card.layout.addLayout(head)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)
        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        for (title, ops), (r, c) in zip(OP_GROUPS, positions):
            group = QWidget()
            group_lay = QVBoxLayout(group)
            group_lay.setContentsMargins(0, 0, 0, 0)
            group_lay.setSpacing(5)
            group_lay.addWidget(label(title, "Caption"))
            flow_host = QWidget()
            flow = FlowLayout(flow_host, 0, 7, 7)
            for op in ops:
                chip = QPushButton(op)
                chip.setProperty("variant", "chip")
                chip.setCheckable(True)
                chip.setCursor(Qt.PointingHandCursor)
                chip.setToolTip(f"{op} · {OP_HINTS.get(op, '')}")
                chip.toggled.connect(self._on_ops_changed)
                self._chips[op] = chip
                flow.addWidget(chip)
            group_lay.addWidget(flow_host)
            grid.addWidget(group, r, c)
        card.layout.addLayout(grid)
        self.ops_hint = label("", "Caption")
        card.layout.addWidget(self.ops_hint)
        return card

    def _build_training_card(self) -> Card:
        card = Card(padding=18)
        card.layout.addWidget(section_title("训练预算", "轮数可自定义到十万轮；自由/固定常量均为无量纲 token"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        epochs_control = QWidget()
        epochs_lay = QHBoxLayout(epochs_control)
        epochs_lay.setContentsMargins(0, 0, 0, 0)
        epochs_lay.setSpacing(5)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setSingleStep(10)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setSuffix(" 轮")
        self.epochs_spin.valueChanged.connect(self._emit_config)
        epochs_lay.addWidget(self.epochs_spin, 1)
        for quick in (100, 1000, 10000):
            btn = QPushButton("1k" if quick == 1000 else ("10k" if quick == 10000 else str(quick)))
            btn.setProperty("variant", "chip")
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda _=False, v=quick: self.epochs_spin.setValue(v))
            epochs_lay.addWidget(btn)
        grid.addWidget(
            ParamTile("迭代轮数", "epochs · 1–100,000，可随时停止", epochs_control), 0, 0
        )

        self.batch_combo = QComboBox()
        for value in [200, 500, 1000, 2000, 4000, 8000]:
            self.batch_combo.addItem(f"{value:,} / epoch", value)
        self.batch_combo.currentIndexChanged.connect(self._emit_config)
        grid.addWidget(
            ParamTile("批大小", "每轮评估的表达式数量；越大越稳、越占资源", self.batch_combo), 0, 1
        )

        self.max_len_spin = QSpinBox()
        self.max_len_spin.setRange(10, 80)
        self.max_len_spin.setValue(20)
        self.max_len_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(
            ParamTile("表达式最大长度", "token 数；越大越能表达复杂关系", self.max_len_spin), 0, 2
        )

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.0001, 0.02)
        self.lr_spin.setSingleStep(0.0005)
        self.lr_spin.setValue(0.0025)
        self.lr_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(ParamTile("学习率", "Adam 优化 RNN 的学习率", self.lr_spin), 1, 0)

        self.stop_spin = QDoubleSpinBox()
        self.stop_spin.setDecimals(5)
        self.stop_spin.setRange(0.90, 0.99999)
        self.stop_spin.setSingleStep(0.0005)
        self.stop_spin.setValue(0.9999)
        self.stop_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(
            ParamTile("提前停止奖励", "SquashedNRMSE，越接近 1 越严格", self.stop_spin), 1, 1
        )

        self.stop_epochs_spin = QSpinBox()
        self.stop_epochs_spin.setRange(0, 30)
        self.stop_epochs_spin.setValue(2)
        self.stop_epochs_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(
            ParamTile("达标后额外轮数", "达到阈值后继续搜索的 epoch 数", self.stop_epochs_spin), 1, 2
        )

        self.free_consts_spin = QSpinBox()
        self.free_consts_spin.setRange(0, 6)
        self.free_consts_spin.setValue(2)
        self.free_consts_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(
            ParamTile("自由常数（无量纲）", "c0, c1… 由 LBFGS 优化并加入表达式", self.free_consts_spin), 2, 0
        )

        fixed_control = QWidget()
        fixed_lay = QHBoxLayout(fixed_control)
        fixed_lay.setContentsMargins(0, 0, 0, 0)
        fixed_lay.setSpacing(4)
        self.fixed_consts_edit = QLineEdit("1, 3.14159")
        self.fixed_consts_edit.setPlaceholderText("例如：1, 3.14159")
        self.fixed_consts_edit.textChanged.connect(self._emit_config)
        fixed_lay.addWidget(self.fixed_consts_edit, 1)
        for text, value in (("π", math.pi), ("e", math.e), ("2", 2.0)):
            btn = QPushButton(text)
            btn.setProperty("variant", "chip")
            btn.setFixedSize(28, 26)
            btn.setToolTip(f"添加无量纲常量 {value:g}")
            btn.clicked.connect(lambda _=False, v=value: self._append_fixed(v))
            fixed_lay.addWidget(btn)
        grid.addWidget(
            ParamTile("固定常数（无量纲）", "1、π、e 等；作为无量纲 token 加入表达式", fixed_control), 2, 1
        )

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(0)
        self.seed_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(ParamTile("随机种子", "固定种子便于复现实验", self.seed_spin), 2, 2)

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU（推荐）", "cpu")
        self.device_combo.addItem("自动", "auto")
        self.device_combo.addItem("CUDA / GPU", "cuda")
        self.device_combo.currentIndexChanged.connect(self._emit_config)
        grid.addWidget(
            ParamTile("计算设备", "PhySO 常受 CPU 限制；GPU 主要加速批量执行", self.device_combo), 3, 0
        )

        self.cpu_threads_spin = QSpinBox()
        self.cpu_threads_spin.setRange(0, 64)
        self.cpu_threads_spin.setSpecialValueText("自动")
        self.cpu_threads_spin.setValue(0)
        self.cpu_threads_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(
            ParamTile("CPU 线程数", "0 为自动；出现 CPU 瓶颈时可手动限制/提高", self.cpu_threads_spin), 3, 1
        )

        self.const_steps_spin = QSpinBox()
        self.const_steps_spin.setRange(1, 20)
        self.const_steps_spin.setValue(6)
        self.const_steps_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(
            ParamTile("常数优化步数", "LBFGS 步数；降低可加速，但可能牺牲常数精度", self.const_steps_spin), 3, 2
        )
        card.layout.addLayout(grid)
        return card

    def _build_advanced_card(self) -> Card:
        card = Card(padding=18)
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.01, 0.5)
        self.risk_spin.setDecimals(3)
        self.risk_spin.setSingleStep(0.01)
        self.risk_spin.setValue(0.05)
        self.risk_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(ParamTile("精英比例", "每轮强化的最优表达式比例", self.risk_spin), 0, 0)

        self.entropy_spin = QDoubleSpinBox()
        self.entropy_spin.setRange(0.0, 0.1)
        self.entropy_spin.setDecimals(3)
        self.entropy_spin.setSingleStep(0.001)
        self.entropy_spin.setValue(0.005)
        self.entropy_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(ParamTile("熵正则", "越大越鼓励探索", self.entropy_spin), 0, 1)

        self.max_eval_spin = QSpinBox()
        self.max_eval_spin.setRange(0, 100000000)
        self.max_eval_spin.setSingleStep(10000)
        self.max_eval_spin.setSpecialValueText("不限制")
        self.max_eval_spin.setValue(0)
        self.max_eval_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(
            ParamTile("最多评估表达式数", "0 表示不限制；可避免长时间运行", self.max_eval_spin), 0, 2
        )

        self.parallel_toggle = ToggleRow(
            "并行奖励计算", "Windows 下建议关闭；序列模式更稳定", checked=False
        )
        self.parallel_toggle.toggle.toggled.connect(self._emit_config)
        grid.addWidget(self.parallel_toggle, 1, 0, 1, 2)

        self.cpu_spin = QSpinBox()
        self.cpu_spin.setRange(1, 32)
        self.cpu_spin.setValue(1)
        self.cpu_spin.setEnabled(False)
        self.cpu_spin.valueChanged.connect(self._emit_config)
        grid.addWidget(ParamTile("并行 CPU 数", "仅在开启并行时生效", self.cpu_spin), 1, 2)
        self.parallel_toggle.toggle.toggled.connect(self.cpu_spin.setEnabled)

        card.layout.addWidget(section_title("高级设置", "通常保持默认即可"))
        card.layout.addWidget(Collapsible("展开高级参数", host, expanded=False))
        return card

    # ------------------------------------------------------------- state IO #
    def _refresh_data(self) -> None:
        prepared = self.state.prepared
        while self.data_badges.count():
            item = self.data_badges.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if prepared is None:
            self.data_name.setText("尚未准备数据")
            self.data_desc.setText("请先到「数据」页导入数据、选择特征并设置量纲。")
            self.data_units.setText("")
            self.start_btn.setEnabled(False)
            return
        self.data_name.setText(prepared.name)
        target = prepared.y_name if prepared.y_name == prepared.y_col else f"{prepared.y_name}（{prepared.y_col}）"
        self.data_desc.setText(f"{prepared.describe()} · 目标变量 {target}")
        unit_parts = [
            f"{name} [{format_units(vec)}]" for name, vec in zip(prepared.x_names, prepared.x_units)
        ]
        unit_parts.append(f"{prepared.y_name} [{format_units(prepared.y_units)}]")
        self.data_units.setText("量纲：" + " · ".join(unit_parts))
        self.data_badges.addWidget(badge(f"{prepared.n_samples} 行", "blue"))
        self.data_badges.addWidget(badge(f"{prepared.n_dim} 维输入", "green"))
        if prepared.n_eval:
            self.data_badges.addWidget(badge(f"评估 {prepared.n_eval}", "orange"))
        if prepared.normalize:
            self.data_badges.addWidget(badge("标准化", "gray"))
        if prepared.use_units and prepared.has_units:
            self.data_badges.addWidget(badge("量纲分析", "blue"))
        self.data_badges.addStretch(1)
        self.start_btn.setEnabled(True)
        self._update_summary()

    def _sync_from_config(self) -> None:
        cfg = self.state.config
        self._syncing = True
        try:
            preset_keys = ["config0", "config2", "config1"]
            self.preset_control.set_current_index(
                preset_keys.index(cfg.preset) if cfg.preset in preset_keys else 0
            )
            self.preset_note.setText(E.PRESETS.get(cfg.preset, E.PRESETS["config0"])["note"])
            self.epochs_spin.setValue(cfg.epochs)
            self.max_len_spin.setValue(cfg.max_length)
            idx = self.batch_combo.findData(cfg.batch_size)
            if idx < 0:
                self.batch_combo.addItem(f"{cfg.batch_size:,} / epoch", cfg.batch_size)
                idx = self.batch_combo.count() - 1
            self.batch_combo.setCurrentIndex(idx)
            self.lr_spin.setValue(cfg.lr)
            self.stop_spin.setValue(cfg.stop_reward)
            self.stop_epochs_spin.setValue(cfg.stop_after_n_epochs)
            self.free_consts_spin.setValue(cfg.n_free_consts)
            self.fixed_consts_edit.setText(", ".join(f"{v:g}" for v in cfg.fixed_consts))
            self.seed_spin.setValue(cfg.seed)
            device_index = self.device_combo.findData(cfg.device)
            self.device_combo.setCurrentIndex(max(0, device_index))
            self.cpu_threads_spin.setValue(cfg.cpu_threads)
            self.const_steps_spin.setValue(cfg.const_opti_steps)
            self.risk_spin.setValue(cfg.risk_factor)
            self.entropy_spin.setValue(cfg.entropy_weight)
            self.parallel_toggle.toggle.setChecked(cfg.parallel)
            self.cpu_spin.setValue(max(1, cfg.n_cpus))
            self.max_eval_spin.setValue(cfg.max_n_evaluations or 0)
            for op, chip in self._chips.items():
                chip.blockSignals(True)
                chip.setChecked(op in cfg.ops)
                chip.blockSignals(False)
        finally:
            self._syncing = False
        self._on_ops_changed()

    def _on_config_external(self) -> None:
        if self._syncing or self._emitting:
            return
        self._sync_from_config()

    def _append_fixed(self, value: float) -> None:
        text = self.fixed_consts_edit.text().strip()
        parts = [part.strip() for part in text.split(",") if part.strip()]
        rendered = f"{value:g}"
        if rendered not in parts:
            parts.append(rendered)
        self.fixed_consts_edit.setText(", ".join(parts))

    def _parse_fixed(self) -> list[float]:
        text = self.fixed_consts_edit.text().strip()
        if not text:
            return []
        values = []
        for token in re.split(r"[,;\s]+", text):
            if token:
                values.append(float(token))
        return values

    def collect_config(self) -> E.TrainConfig:
        return E.TrainConfig(
            preset=["config0", "config2", "config1"][max(0, self.preset_control.current_index())],
            epochs=int(self.epochs_spin.value()),
            batch_size=int(self.batch_combo.currentData() or 1000),
            lr=float(self.lr_spin.value()),
            max_length=int(self.max_len_spin.value()),
            stop_reward=float(self.stop_spin.value()),
            stop_after_n_epochs=int(self.stop_epochs_spin.value()),
            max_n_evaluations=int(self.max_eval_spin.value()) or None,
            seed=int(self.seed_spin.value()),
            device=str(self.device_combo.currentData() or "auto"),
            parallel=bool(self.parallel_toggle.isChecked()),
            ops=[op for op, chip in self._chips.items() if chip.isChecked()],
            n_free_consts=int(self.free_consts_spin.value()),
            fixed_consts=self._parse_fixed(),
            risk_factor=float(self.risk_spin.value()),
            entropy_weight=float(self.entropy_spin.value()),
            n_cpus=int(self.cpu_spin.value()),
            cpu_threads=int(self.cpu_threads_spin.value()),
            const_opti_steps=int(self.const_steps_spin.value()),
        )

    def _emit_config(self, *_args) -> None:
        if self._syncing:
            return
        try:
            cfg = self.collect_config()
        except ValueError:
            self.ops_hint.setText("固定常数格式无效，请用逗号分隔数字。")
            return
        self._emitting = True
        try:
            self.state.set_config(cfg)
        finally:
            self._emitting = False
        self._update_summary()

    def _update_summary(self) -> None:
        ops = sum(1 for chip in self._chips.values() if chip.isChecked())
        if not ops:
            self.ops_hint.setText("尚未选择算子，训练前至少需要一个。")
        else:
            self.ops_hint.setText(f"已选择 {ops} 个算子 · 最大长度 {self.max_len_spin.value()}")
        self.summary_label.setText(
            f"{self.epochs_spin.value():,} 轮 · batch {int(self.batch_combo.currentData() or 0):,} · "
            f"CPU 线程 {self.cpu_threads_spin.value() or '自动'}"
        )

    def _on_section_changed(self, index: int) -> None:
        self.section_stack.setCurrentIndex(index)

    def show_section(self, index: int) -> None:
        index = max(0, min(index, self.section_stack.count() - 1))
        self.section_switch.set_current_index(index)
        self.section_stack.setCurrentIndex(index)

    def _on_preset(self, index: int) -> None:
        key = ["config0", "config2", "config1"][index]
        self.preset_note.setText(E.PRESETS[key]["note"])
        batch = {"config0": 500, "config2": 1000, "config1": 2000}[key]
        idx = self.batch_combo.findData(batch)
        if idx >= 0:
            self.batch_combo.setCurrentIndex(idx)
        self._emit_config()

    def _on_ops_changed(self, *_args) -> None:
        if self._syncing:
            return
        self._update_summary()
        self._emit_config()

    def _set_ops(self, ops: list[str]) -> None:
        self._syncing = True
        try:
            for name, chip in self._chips.items():
                chip.setChecked(name in ops)
        finally:
            self._syncing = False
        self._on_ops_changed()

    def _select_recommended(self) -> None:
        self._set_ops(E.DEFAULT_OPS)

    def set_environment(self, info: dict) -> None:
        if info.get("cuda") in {"CPU only", "不可用"}:
            self.env_label.setText("CPU 模式")
            self.env_detail.setText(
                f"Python {info.get('python')} · PyTorch {info.get('torch')}\n"
                "未检测到 CUDA，训练仍可进行。"
            )
        else:
            version = str(info.get("torch", "")).split("+")[0]
            self.env_label.setText(f"CUDA {info.get('cuda')} · PyTorch {version}")
            self.env_detail.setText(f"{info.get('gpu', 'GPU')}\nGPU 利用率低并不代表训练异常。")

    def _on_start(self) -> None:
        if self.state.prepared is None:
            self.toast("请先在「数据」页准备训练数据。", "error")
            return
        if not any(chip.isChecked() for chip in self._chips.values()):
            self.toast("至少选择一个符号算子。", "error")
            return
        try:
            cfg = self.collect_config()
        except ValueError as exc:
            self.toast(f"固定常数格式无效：{exc}", "error")
            return
        if cfg.max_length < 10:
            self.toast("表达式最大长度至少为 10。", "error")
            return
        self.state.set_config(cfg)
        self.start_requested.emit()
