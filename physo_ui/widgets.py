"""Reusable iOS-style widgets used across the application."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from . import icons, theme


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
class FlowLayout(QLayout):
    """A wrapping layout for chips and tags."""

    def __init__(self, parent: QWidget | None = None, margin: int = 0, hspacing: int = 8, vspacing: int = 8):
        super().__init__(parent)
        self._items: list[QWidgetItem] = []
        self._hspace = hspacing
        self._vspace = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._do_layout(QRectF(0, 0, width, 0), True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(QRectF(rect), False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRectF, test_only: bool) -> int:
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self._hspace
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + self._vspace
                next_x = x + hint.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRectF(x, y, hint.width(), hint.height()).toRect())
            x = next_x
            line_height = max(line_height, hint.height())
        return int(y + line_height - rect.y())


def clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget() if item is not None else None
        if widget is not None:
            widget.deleteLater()


def add_shadow(widget: QWidget, blur: int = 28, dy: int = 6, alpha: int = 22) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, dy)
    effect.setColor(QColor(20, 20, 40, alpha))
    widget.setGraphicsEffect(effect)


# --------------------------------------------------------------------------- #
# Basic containers
# --------------------------------------------------------------------------- #
class Card(QFrame):
    def __init__(self, parent: QWidget | None = None, *, padding: int = 20, shadow: bool = True, object_name: str = "Card"):
        super().__init__(parent)
        self.setObjectName(object_name)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(12)
        if shadow:
            add_shadow(self, blur=26, dy=6, alpha=14)

    @property
    def layout(self) -> QVBoxLayout:  # type: ignore[override]
        return self._layout


class InnerCard(QFrame):
    def __init__(self, parent: QWidget | None = None, *, padding: int = 14):
        super().__init__(parent)
        self.setObjectName("InnerCard")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(8)

    @property
    def layout(self) -> QVBoxLayout:  # type: ignore[override]
        return self._layout


class GradientCard(QFrame):
    """Hero panel with a painted iOS gradient (independent of QSS parsing)."""

    def __init__(self, parent=None, *, radius: int = 20):
        super().__init__(parent)
        self.setObjectName("HeroCard")
        self._radius = radius
        self.setAttribute(Qt.WA_StyledBackground, False)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(theme.BLUE))
        gradient.setColorAt(1.0, QColor("#5E5CE6"))
        p.setPen(Qt.NoPen)
        p.setBrush(gradient)
        p.drawRoundedRect(self.rect().adjusted(0, 0, -0.5, -0.5), self._radius, self._radius)
        p.end()


class HLine(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFixedHeight(1)


def vspace(height: int) -> QSpacerItem:
    return QSpacerItem(0, height, QSizePolicy.Minimum, QSizePolicy.Fixed)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def label(text: str, object_name: str = "", *, wrap: bool = False, selectable: bool = False) -> QLabel:
    lbl = QLabel(text)
    if object_name:
        lbl.setObjectName(object_name)
    lbl.setWordWrap(wrap)
    lbl.setTextInteractionFlags(
        Qt.TextSelectableByMouse if selectable else Qt.LinksAccessibleByMouse
    )
    return lbl


def section_title(text: str, subtitle: str | None = None) -> QWidget:
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    lay.addWidget(label(text, "SectionTitle"))
    if subtitle:
        lay.addWidget(label(subtitle, "Caption", wrap=True))
    return box


def badge(text: str, kind: str = "blue") -> QLabel:
    names = {"blue": "Badge", "green": "BadgeGreen", "orange": "BadgeOrange", "gray": "BadgeGray"}
    lbl = QLabel(text)
    lbl.setObjectName(names.get(kind, "Badge"))
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return lbl


class MetricTile(Card):
    def __init__(self, title: str, value: str = "—", hint: str = "", accent: str = theme.BLUE, parent=None):
        super().__init__(parent, padding=16)
        self.setMinimumHeight(104)
        self._accent = QLabel()
        self._accent.setFixedSize(4, 18)
        self._accent.setStyleSheet(f"background:{accent}; border-radius:2px;")
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._accent)
        self.title_label = label(title, "MetricTitle")
        top.addWidget(self.title_label)
        top.addStretch(1)
        self.value_label = label(value, "MetricValue")
        self.hint_label = label(hint, "Caption")
        self.layout.addLayout(top)
        self.layout.addWidget(self.value_label)
        self.layout.addWidget(self.hint_label)

    def set_value(self, value: str, hint: str | None = None) -> None:
        self.value_label.setText(value)
        if hint is not None:
            self.hint_label.setText(hint)


class EmptyState(QWidget):
    def __init__(self, title: str, text: str, icon_kind: str = "sparkles", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 36, 24, 36)
        lay.setSpacing(10)
        lay.addStretch(1)
        mark = _IconMark(icon_kind, 52, theme.BLUE, theme.BLUE_SOFT)
        lay.addWidget(mark, alignment=Qt.AlignHCenter)
        lay.addSpacing(4)
        lay.addWidget(label(title, "EmptyTitle"), alignment=Qt.AlignHCenter)
        txt = label(text, "EmptyText", wrap=True)
        txt.setAlignment(Qt.AlignHCenter)
        lay.addWidget(txt, alignment=Qt.AlignHCenter)
        self.actions = QVBoxLayout()
        self.actions.setContentsMargins(0, 6, 0, 0)
        self.actions.setSpacing(8)
        lay.addLayout(self.actions)
        lay.addStretch(1)

    def add_action(self, widget: QWidget, alignment=Qt.AlignHCenter) -> None:
        self.actions.addWidget(widget, alignment=alignment)

    def add_action_layout(self, layout: QLayout) -> None:
        self.actions.addLayout(layout)


class _IconMark(QWidget):
    def __init__(self, kind: str, size: int, color: str, background: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.color = color
        self.background = background
        self.setFixedSize(size, size)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.background))
        p.drawRoundedRect(self.rect(), 14, 14)
        icons.paint_icon(p, self.kind, QRectF(12, 12, self.width() - 24, self.height() - 24), self.color)
        p.end()


class IconBadge(QWidget):
    def __init__(self, kind: str, color: str = theme.BLUE, background: str = theme.BLUE_SOFT, size: int = 34, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.color = color
        self.background = background
        self.setFixedSize(size, size)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.background))
        p.drawRoundedRect(self.rect(), 10, 10)
        pad = self.width() * 0.24
        icons.paint_icon(p, self.kind, QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad), self.color)
        p.end()


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
class Toggle(QAbstractButton):
    """An iOS-style switch."""

    def __init__(self, parent: QWidget | None = None, *, checked: bool = False):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(48, 30)
        self._position = 1.0 if checked else 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.toggled.connect(self._animate)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(48, 30)

    def _on_anim(self, value) -> None:
        self._position = float(value)
        self.update()

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._position)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        track_on = QColor(theme.GREEN if self.isEnabled() else "#B9E7C4")
        track_off = QColor(theme.FILL_2)
        t = self._position
        color = QColor(
            int(track_off.red() + (track_on.red() - track_off.red()) * t),
            int(track_off.green() + (track_on.green() - track_off.green()) * t),
            int(track_off.blue() + (track_on.blue() - track_off.blue()) * t),
        )
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(QRectF(0, 0, 48, 30), 15, 15)
        knob_x = 3 + t * 21
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QRectF(knob_x, 3, 24, 24))
        p.end()


class SegmentedControl(QWidget):
    currentChanged = Signal(int)

    def __init__(self, items: Iterable[str], parent: QWidget | None = None, *, current: int = 0):
        super().__init__(parent)
        self.setObjectName("Segmented")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"""
            QWidget#Segmented {{ background: {theme.FILL_2}; border-radius: 10px; }}
            QPushButton[variant="segment"] {{
                background: transparent; color: {theme.INK_2}; border: none;
                border-radius: 8px; padding: 6px 14px; font-size: 12px; font-weight: 600;
            }}
            QPushButton[variant="segment"]:hover {{ color: {theme.INK}; }}
            QPushButton[variant="segment"]:checked {{ background: white; color: {theme.INK}; }}
            """
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QPushButton] = []
        for i, text in enumerate(items):
            btn = QPushButton(text)
            btn.setProperty("variant", "segment")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            self._group.addButton(btn, i)
            lay.addWidget(btn)
            self._buttons.append(btn)
        if self._buttons:
            self._buttons[max(0, min(current, len(self._buttons) - 1))].setChecked(True)
        self._group.idClicked.connect(self.currentChanged.emit)

    def current_index(self) -> int:
        return self._group.checkedId()

    def set_current_index(self, index: int) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)


class LabeledSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(self, title: str, minimum: int, maximum: int, value: int, *, suffix: str = "", step: int = 1, parent=None):
        super().__init__(parent)
        self._suffix = suffix
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.title = label(title, "Caption")
        self.value_label = label("", "Badge")
        self.value_label.setMinimumWidth(54)
        self.value_label.setAlignment(Qt.AlignCenter)
        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.value_label)
        lay.addLayout(top)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(max(step, (maximum - minimum) // 10))
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._on_change)
        lay.addWidget(self.slider)
        self._on_change(value)

    def _on_change(self, value: int) -> None:
        self.value_label.setText(f"{value}{self._suffix}")
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int) -> None:
        self.slider.setValue(value)


class ToggleRow(QWidget):
    def __init__(self, title: str, subtitle: str = "", *, checked: bool = False, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(2)
        texts.addWidget(label(title, "CardTitle"))
        if subtitle:
            texts.addWidget(label(subtitle, "Caption", wrap=True))
        lay.addLayout(texts, 1)
        self.toggle = Toggle(checked=checked)
        lay.addWidget(self.toggle, alignment=Qt.AlignVCenter)

    def isChecked(self) -> bool:  # noqa: N802
        return self.toggle.isChecked()


class FieldRow(QWidget):
    """Label on the left, arbitrary control on the right."""

    def __init__(self, title: str, control: QWidget, subtitle: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(2)
        texts.addWidget(label(title, "CardTitle"))
        if subtitle:
            c = label(subtitle, "Caption", wrap=True)
            texts.addWidget(c)
        lay.addLayout(texts, 1)
        control.setMinimumWidth(130)
        lay.addWidget(control, 0, Qt.AlignVCenter)


class InfoRow(QWidget):
    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(10)
        self.title = label(title, "Caption")
        self.value = label(value, "CardTitle", selectable=True)
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.title)
        lay.addStretch(1)
        lay.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


class Collapsible(QWidget):
    def __init__(self, title: str, content: QWidget, *, expanded: bool = False, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = expanded
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.button = QPushButton()
        self.button.setProperty("variant", "ghost")
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setStyleSheet("text-align:left; padding:6px 2px; font-weight:600;")
        lay.addWidget(self.button)
        self.content = content
        lay.addWidget(content)
        self.button.clicked.connect(self.toggle)
        self._refresh()

    def _refresh(self) -> None:
        self.button.setText(("▾  " if self._expanded else "▸  ") + self._title)
        self.content.setVisible(self._expanded)

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._refresh()


class ScrollPage(QWidget):
    """Page with a large iOS title and a scrollable body."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(34, 26, 34, 0)
        outer.setSpacing(16)
        self.title_label = label(title, "PageTitle")
        self.subtitle_label = label(subtitle, "PageSubtitle", wrap=True)
        outer.addWidget(self.title_label)
        if subtitle:
            outer.addWidget(self.subtitle_label)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("PageScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.viewport().setAutoFillBackground(False)
        self.body = QWidget()
        self.body.setObjectName("ScrollBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 4, 8, 30)
        self.body_layout.setSpacing(16)
        self.scroll.setWidget(self.body)
        outer.addWidget(self.scroll, 1)

    def add(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def add_layout(self, layout: QLayout) -> None:
        self.body_layout.addLayout(layout)

    def add_stretch(self, stretch: int = 1) -> None:
        self.body_layout.addStretch(stretch)


class LoadingDots(QLabel):
    def __init__(self, parent=None):
        super().__init__("", parent)
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setStyleSheet(f"color:{theme.GRAY};")
        self._timer.start(350)

    def _tick(self) -> None:
        self._dots = (self._dots + 1) % 4
        self.setText("处理中" + "·" * self._dots)

    def stop(self) -> None:
        self._timer.stop()


class Spinner(QWidget):
    """Simple rotating arc, used while the engine starts."""

    def __init__(self, size: int = 28, color: str = theme.BLUE, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._color = QColor(color)
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    def _tick(self) -> None:
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(self._color, 3)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        margin = 4
        p.drawArc(self.rect().adjusted(margin, margin, -margin, -margin), self._angle * 16, 260 * 16)
        p.end()

    def stop(self) -> None:
        self._timer.stop()


class VerticalScrollArea(QScrollArea):
    def __init__(self, widget: QWidget, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(widget)


def dot(color: str, size: int = 8) -> QLabel:
    lbl = QLabel()
    lbl.setFixedSize(size, size)
    lbl.setStyleSheet(f"background:{color}; border-radius:{size // 2}px;")
    return lbl
