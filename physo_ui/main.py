"""Main window, rounded iOS-style chrome, navigation and application bootstrap."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QRegion,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

from . import data as D
from . import engine as E
from . import icons, theme
from .pages.data_page import DataPage
from .pages.model_page import ModelPage
from .pages.predict_page import PredictPage
from .pages.result_page import ResultPage
from .pages.train_page import TrainPage
from .state import AppState
from .widgets import label


PAGE_NAMES = ["数据", "模型", "训练", "结果", "推理"]


class LogoMark(QWidget):
    def __init__(self, size: int = 42, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(theme.BLUE))
        gradient.setColorAt(1, QColor("#5E5CE6"))
        p.setPen(Qt.NoPen)
        p.setBrush(gradient)
        p.drawRoundedRect(self.rect(), max(6, self.width() // 4), max(6, self.width() // 4))
        p.setPen(QColor("#FFFFFF"))
        font = QFont(theme.app_font(14))
        font.setBold(True)
        font.setPointSize(max(10, int(self.width() * 0.45)))
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignCenter, "Φ")
        p.end()


class Toast(QLabel):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMaximumWidth(640)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)
        self._fading_out = False
        self._anim.finished.connect(self._on_anim_finished)
        self.hide()

    def show_message(self, text: str, kind: str = "info") -> None:
        colors = {
            "success": theme.GREEN,
            "error": theme.RED,
            "warn": theme.ORANGE,
            "info": theme.BLUE,
        }
        self.setStyleSheet(
            f"background:{colors.get(kind, theme.BLUE)}; color:white; border-radius:12px;"
            "padding:10px 18px; font-size:13px; font-weight:600;"
        )
        self.setText(text)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self._anim.stop()
        self._anim.setStartValue(self._effect.opacity())
        self._anim.setEndValue(1.0)
        self._anim.start()
        self._timer.start(2600)

    def _fade_out(self) -> None:
        self._anim.stop()
        self._fading_out = True
        self._anim.setStartValue(self._effect.opacity())
        self._anim.setEndValue(0.0)
        self._anim.start()

    def _on_anim_finished(self) -> None:
        if self._fading_out and self._effect.opacity() <= 0.01:
            self.hide()
            self._fading_out = False

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        x = (parent.width() - self.width()) // 2
        y = parent.height() - self.height() - 38
        self.move(max(16, x), max(16, y))


class ResizeGrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.SizeFDiagCursor)
        self.setToolTip("拖动调整窗口大小")

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QColor(150, 150, 160, 150))
        for offset in (4, 9, 14):
            p.drawLine(20 - offset, 18, 18, 20 - offset)
        p.end()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemResize(Qt.BottomEdge | Qt.RightEdge)


class WindowFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WindowFrame")
        self._rounded = True
        self.grip = ResizeGrip(self)

    def set_rounded(self, rounded: bool) -> None:
        self._rounded = rounded
        self.update_mask()

    def update_mask(self) -> None:
        if not self._rounded or self.width() <= 1 or self.height() <= 1:
            self.clearMask()
            return
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 22, 22)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self.grip.move(self.width() - 22, self.height() - 22)
        self.grip.raise_()
        self.update_mask()


class TitleBar(QFrame):
    def __init__(self, window: "MainWindow", parent=None):
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self.setFixedHeight(48)
        self._window = window
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 6, 10, 6)
        lay.setSpacing(10)
        lay.addWidget(LogoMark(28))
        texts = QVBoxLayout()
        texts.setSpacing(0)
        texts.addWidget(label("PhySO Studio", "WindowTitle"))
        texts.addWidget(label("物理符号回归", "WindowSub"))
        lay.addLayout(texts)
        lay.addStretch(1)
        self.status = label("●  本地运行", "Caption")
        self.status.setStyleSheet(f"color:{theme.GREEN}; font-size:11px;")
        lay.addWidget(self.status)
        lay.addSpacing(8)
        self.min_btn = self._window_button("minimize", "最小化")
        self.max_btn = self._window_button("maximize", "最大化")
        self.close_btn = self._window_button("close", "关闭")
        self.close_btn.setObjectName("WinClose")
        lay.addWidget(self.min_btn)
        lay.addWidget(self.max_btn)
        lay.addWidget(self.close_btn)

    def _window_button(self, kind: str, tooltip: str) -> QPushButton:
        btn = QPushButton()
        btn.setProperty("variant", "win")
        btn.setIcon(icons.icon(kind, theme.INK_2, 14))
        btn.setIconSize(QSize(14, 14))
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemMove()

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._window.toggle_maximized()


class EnvProbe(QThread):
    ready = Signal(dict)

    def run(self) -> None:
        try:
            info = E.environment_info()
        except Exception:
            info = {}
        self.ready.emit(info)


class MainWindow(QMainWindow):
    def __init__(self, *, screenshot_all: str | None = None, smoke: bool = False):
        super().__init__()
        self.setWindowTitle("PhySO Studio · 物理符号回归")
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRectF(0, 0, 1440, 860)
        width = min(1440, max(1080, int(available.width()) - 40))
        height = min(900, max(660, int(available.height())))
        self.setMinimumSize(min(1080, width), min(660, height))
        self.resize(width, height)
        if screen is not None:
            self.move(
                int(available.center().x() - width / 2),
                int(available.center().y() - height / 2),
            )
        self.setWindowIcon(make_app_icon())
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.state = AppState()
        self._smoke = smoke
        self._screenshot_all = Path(screenshot_all) if screenshot_all else None
        self._shadow: QGraphicsDropShadowEffect | None = None
        self._build()
        self._toast = Toast(self)
        self._probe = EnvProbe(self)
        self._probe.ready.connect(self._on_env_ready)
        self._probe.start()
        self._apply_window_state()

    # ------------------------------------------------------------------ UI #
    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        root.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCentralWidget(root)
        self._root_layout = QVBoxLayout(root)
        self._root_layout.setContentsMargins(16, 16, 16, 16)
        self._root_layout.setSpacing(0)

        self.frame = WindowFrame()
        self._root_layout.addWidget(self.frame)
        self.frame_layout = QVBoxLayout(self.frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(0)

        self.titlebar = TitleBar(self)
        self.titlebar.min_btn.clicked.connect(self.showMinimized)
        self.titlebar.max_btn.clicked.connect(self.toggle_maximized)
        self.titlebar.close_btn.clicked.connect(self.close)
        self.frame_layout.addWidget(self.titlebar)

        content = QWidget()
        content.setObjectName("ContentShell")
        content.setAttribute(Qt.WA_TranslucentBackground, True)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._build_sidebar())
        self.stack = QStackedWidget()
        self.stack.setObjectName("Stack")
        content_layout.addWidget(self.stack, 1)
        self.frame_layout.addWidget(content, 1)

        self.data_page = DataPage(self.state, self.show_toast)
        self.model_page = ModelPage(self.state, self.show_toast)
        self.train_page = TrainPage(self.state, self.show_toast)
        self.result_page = ResultPage(self.state, self.show_toast)
        self.predict_page = PredictPage(self.state, self.show_toast)
        self.pages = [
            self.data_page,
            self.model_page,
            self.train_page,
            self.result_page,
            self.predict_page,
        ]
        for page in self.pages:
            self.stack.addWidget(page)

        self.data_page.continue_requested.connect(lambda: self.go_to(1))
        self.data_page.settings_requested.connect(self._show_model_constants)
        self.model_page.edit_data_requested.connect(lambda: self.go_to(0))
        self.model_page.start_requested.connect(lambda: self.go_to(2))
        self.train_page.data_requested.connect(lambda: self.go_to(0))
        self.train_page.result_requested.connect(lambda: self.go_to(3))
        self.result_page.train_requested.connect(lambda: self.go_to(2))
        self.result_page.predict_requested.connect(lambda: self.go_to(4))
        self.predict_page.train_requested.connect(lambda: self.go_to(2))
        self.predict_page.result_requested.connect(lambda: self.go_to(3))

        for i in range(len(self.pages)):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            shortcut.activated.connect(lambda idx=i: self.go_to(idx))
        self.go_to(0, animate=False)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(14, 18, 14, 16)
        lay.setSpacing(5)
        lay.addWidget(label("工作流", "Caption"))
        lay.addSpacing(2)

        self.nav_buttons: list[QPushButton] = []
        self._nav_items = [
            ("sparkles", "数据"),
            ("model", "模型"),
            ("train", "训练"),
            ("result", "结果"),
            ("predict", "推理"),
        ]
        for i, (kind, text) in enumerate(self._nav_items):
            btn = QPushButton(text)
            btn.setProperty("variant", "navItem")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(44)
            btn.setIconSize(QSize(20, 20))
            btn.setIcon(icons.icon(kind, theme.GRAY, 20))
            btn.clicked.connect(lambda checked=False, idx=i: self.go_to(idx))
            lay.addWidget(btn)
            self.nav_buttons.append(btn)
        lay.addSpacing(14)
        lay.addStretch(1)

        env = QFrame()
        env.setObjectName("InnerCard")
        env_lay = QVBoxLayout(env)
        env_lay.setContentsMargins(14, 11, 14, 11)
        env_lay.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(6)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{theme.GREEN}; border-radius:4px;")
        head.addWidget(dot)
        head.addWidget(label("本地运行", "Caption"))
        head.addStretch(1)
        env_lay.addLayout(head)
        self.env_line = label("正在检测环境…", "Caption", wrap=True)
        env_lay.addWidget(self.env_line)
        self.physo_line = label("", "Caption", wrap=True)
        env_lay.addWidget(self.physo_line)
        lay.addWidget(env)
        lay.addSpacing(6)
        lay.addWidget(label("PhySO Studio v1.0 · MIT", "Caption"))
        return sidebar

    def _show_model_constants(self) -> None:
        self.go_to(1)
        self.model_page.show_section(1)

    # -------------------------------------------------------------- navigation #
    def go_to(self, index: int, animate: bool = True) -> None:
        if not (0 <= index < len(self.pages)):
            return
        self.stack.setCurrentIndex(index)
        page = self.pages[index]
        scroll = getattr(page, "scroll", None)
        if scroll is not None:
            scroll.verticalScrollBar().setValue(0)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
            kind = self._nav_items[i][0]
            btn.setIcon(icons.icon(kind, theme.BLUE_DARK if i == index else theme.GRAY, 20))
        if index == 1:
            self.model_page._refresh_data()
        elif index == 2:
            self.train_page._refresh_summary()
        elif index == 3:
            self.result_page.refresh()
        elif index == 4:
            self.predict_page.refresh()
        if animate and not self._smoke:
            self._fade_page(self.pages[index])

    def _fade_page(self, page: QWidget) -> None:
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", page)
        anim.setDuration(160)
        anim.setStartValue(0.6)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.finished.connect(lambda: page.setGraphicsEffect(None))
        anim.start()
        page._fade_anim = anim

    # ---------------------------------------------------------------- window #
    def toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _apply_window_state(self) -> None:
        maximized = self.isMaximized()
        if maximized:
            self._root_layout.setContentsMargins(0, 0, 0, 0)
        else:
            self._root_layout.setContentsMargins(16, 16, 16, 16)
        self.frame.setProperty("maximized", "true" if maximized else "false")
        self.frame.style().unpolish(self.frame)
        self.frame.style().polish(self.frame)
        self.frame.grip.setVisible(not maximized)
        self.frame.set_rounded(not maximized)
        if maximized:
            self.frame.setGraphicsEffect(None)
            self._shadow = None
        else:
            self._install_shadow()

    def _install_shadow(self) -> None:
        if self._shadow is not None:
            return
        shadow = QGraphicsDropShadowEffect(self.frame)
        shadow.setBlurRadius(54)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(20, 20, 40, 70))
        self.frame.setGraphicsEffect(shadow)
        self._shadow = shadow

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange and hasattr(self, "frame"):
            self._apply_window_state()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._apply_window_state()

    # ---------------------------------------------------------------- utility #
    def show_toast(self, text: str, kind: str = "info") -> None:
        self._toast.show_message(text, kind)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_toast"):
            self._toast._reposition()

    def _on_env_ready(self, info: dict) -> None:
        if not info:
            self.env_line.setText("环境检测失败")
            return
        if info.get("cuda") in {"CPU only", "不可用"}:
            self.env_line.setText(f"CPU · Python {info.get('python')}")
        else:
            version = str(info.get("torch", "")).split("+")[0]
            self.env_line.setText(f"CUDA {info.get('cuda')} · PyTorch {version}")
        self.physo_line.setText(f"PhySO {info.get('physo')}")
        self.model_page.set_environment(info)

    def closeEvent(self, event):  # noqa: N802
        worker = self.state.worker
        if worker is not None and worker.isRunning():
            answer = QMessageBox.question(
                self,
                "训练仍在进行",
                "停止训练并退出？当前 epoch 结束后会保存已找到的结果。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            worker.request_stop()
            self.show_toast("正在结束当前 epoch…", "warn")
            if not worker.wait(8000):
                QMessageBox.information(self, "请稍候", "训练仍在结束中，请稍后再关闭窗口。")
                event.ignore()
                return
        if hasattr(self, "_probe") and self._probe.isRunning():
            self._probe.wait(5000)
        super().closeEvent(event)

    # -------------------------------------------------------------- automation #
    def run_screenshot(self, path: str, delay_ms: int = 4200) -> None:
        def capture() -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.grab().save(path)
            print(f"Screenshot saved to {path}")
            self.close()
            QApplication.instance().quit()

        QTimer.singleShot(delay_ms, capture)

    def run_screenshot_all(self, folder: str, delay_ms: int = 4000) -> None:
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        def capture(index: int = 0) -> None:
            if index >= len(self.pages):
                print(f"Screenshots saved to {folder_path}")
                self.close()
                QApplication.instance().quit()
                return
            self.go_to(index, animate=False)
            QApplication.processEvents()
            self.grab().save(str(folder_path / f"{index + 1:02d}-{PAGE_NAMES[index]}.png"))
            QTimer.singleShot(450, lambda: capture(index + 1))

        QTimer.singleShot(delay_ms, lambda: capture(0))

    def run_smoke(self) -> None:
        def step(index: int = 0) -> None:
            if index >= len(self.pages):
                print("GUI SMOKE OK")
                self.close()
                QApplication.instance().quit()
                return
            self.go_to(index, animate=False)
            QApplication.processEvents()
            QTimer.singleShot(180, lambda: step(index + 1))

        QTimer.singleShot(700, lambda: step(0))


def make_app_icon() -> QIcon:
    from PySide6.QtGui import QPixmap

    pm = QPixmap(256, 256)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    gradient = QLinearGradient(0, 0, 256, 256)
    gradient.setColorAt(0, QColor(theme.BLUE))
    gradient.setColorAt(1, QColor("#5E5CE6"))
    p.setPen(Qt.NoPen)
    p.setBrush(gradient)
    p.drawRoundedRect(QRectF(8, 8, 240, 240), 56, 56)
    p.setPen(QColor("white"))
    font = QFont(theme.app_font(120))
    font.setBold(True)
    p.setFont(font)
    p.drawText(QRectF(8, 8, 240, 240), Qt.AlignCenter, "Φ")
    p.end()
    return QIcon(pm)


def _load_demo_into(app: MainWindow, filename: str, prepare: bool) -> None:
    try:
        raw = D.load_bundled(filename)
    except Exception as exc:
        print(f"Unable to load demo {filename}: {exc}", file=sys.stderr)
        return
    app.data_page._meta = app.data_page._library.get(filename, {})
    app.state.set_raw(raw)
    QApplication.processEvents()
    if prepare:
        app.data_page._apply()
        QApplication.processEvents()


def _load_result_into(app: MainWindow, path: str) -> None:
    try:
        result = E.RunResult.load(path)
    except Exception as exc:
        print(f"Unable to load result {path}: {exc}", file=sys.stderr)
        return
    app.state.set_result(result)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="PhySO Studio — iOS-style symbolic regression GUI")
    parser.add_argument("--page", default=None, choices=["data", "model", "train", "result", "predict"])
    parser.add_argument("--demo", default=None, help="bundled dataset filename, e.g. pendulum_period.csv")
    parser.add_argument("--prepare", action="store_true", help="prepare the demo dataset for training")
    parser.add_argument("--load-result", default=None, help="path to a result.json")
    parser.add_argument("--screenshot", default=None, help="save a screenshot and quit")
    parser.add_argument("--screenshot-all", default=None, help="save one screenshot per page and quit")
    parser.add_argument("--smoke", action="store_true", help="open every page, then quit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    app = QApplication(sys.argv[:1])
    app.setApplicationName("PhySO Studio")
    app.setOrganizationName("PhySO-UI")
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(theme.app_font(10))
    app.setStyleSheet(theme.STYLESHEET)
    app.setWindowIcon(make_app_icon())

    window = MainWindow(screenshot_all=args.screenshot_all, smoke=args.smoke)
    window.show()

    if args.load_result:
        _load_result_into(window, args.load_result)
    if args.demo:
        _load_demo_into(window, args.demo, args.prepare)

    page_map = {"data": 0, "model": 1, "train": 2, "result": 3, "predict": 4}
    if args.page:
        window.go_to(page_map[args.page], animate=False)

    if args.screenshot_all:
        window.run_screenshot_all(args.screenshot_all)
    elif args.screenshot:
        window.run_screenshot(args.screenshot)
    elif args.smoke:
        window.run_smoke()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
