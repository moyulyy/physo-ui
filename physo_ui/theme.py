"""iOS-flavoured design tokens and Qt stylesheet for PhySO Studio."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase

# ---- iOS system colours (light appearance) ---------------------------------
BLUE = "#007AFF"
BLUE_DARK = "#0062CC"
BLUE_SOFT = "#E8F1FF"
GREEN = "#34C759"
GREEN_SOFT = "#E9F9EE"
ORANGE = "#FF9500"
ORANGE_SOFT = "#FFF4E5"
RED = "#FF3B30"
RED_SOFT = "#FFECEB"
PURPLE = "#5856D6"
PURPLE_SOFT = "#EFEEFF"

INK = "#1C1C1E"
INK_2 = "#3A3A3C"
GRAY = "#8E8E93"
GRAY_2 = "#AEAEB2"
SEPARATOR = "#E5E5EA"
BG = "#F2F2F7"
CARD = "#FFFFFF"
CARD_ALT = "#F9F9FB"
FILL = "#F2F2F7"
FILL_2 = "#E9E9EF"

FONT_FAMILIES = [
    "Segoe UI Variable Text",
    "Segoe UI",
    "Microsoft YaHei UI",
    "PingFang SC",
    "Arial",
]


def _installed_families() -> set[str]:
    try:
        return set(QFontDatabase().families())
    except Exception:
        return set()


def app_font(point_size: int = 10) -> QFont:
    font = QFont()
    installed = _installed_families()
    font.setFamilies([f for f in FONT_FAMILIES if f in installed] or FONT_FAMILIES)
    font.setPointSize(point_size)
    font.setHintingPreference(QFont.PreferFullHinting)
    return font


def mono_font(point_size: int = 10) -> QFont:
    families = ["Cascadia Code", "JetBrains Mono", "Consolas", "Courier New"]
    installed = _installed_families()
    available = [f for f in families if f in installed]
    font = QFont()
    font.setFamilies(available or families)
    font.setPointSize(point_size)
    return font


def qcolor(value: str, alpha: int | None = None) -> QColor:
    c = QColor(value)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


STYLESHEET = f"""
* {{
    outline: none;
}}
QWidget {{
    color: {INK};
    font-size: 14px;
    selection-background-color: {BLUE_SOFT};
    selection-color: {INK};
}}
QMainWindow, QWidget#Root {{
    background: transparent;
}}
QWidget#Page, QScrollArea#PageScroll, QWidget#ScrollBody, QStackedWidget#Stack {{
    background: transparent;
}}
QFrame#WindowFrame {{
    background: {BG};
    border: 1px solid rgba(60, 60, 67, 0.30);
    border-radius: 22px;
}}
QFrame#WindowFrame[maximized="true"] {{
    border-radius: 0px;
    border: none;
}}
QFrame#TitleBar {{
    background: transparent;
    border-bottom: 1px solid rgba(60, 60, 67, 0.10);
}}
QLabel#WindowTitle {{
    font-size: 13px;
    font-weight: 700;
    color: {INK};
}}
QLabel#WindowSub {{
    font-size: 10px;
    color: {GRAY};
}}
QPushButton[variant="win"] {{
    background: transparent;
    border: none;
    border-radius: 9px;
    padding: 0;
    min-width: 30px;
    max-width: 30px;
    min-height: 30px;
    max-height: 30px;
}}
QPushButton[variant="win"]:hover {{
    background: rgba(118, 118, 128, 0.14);
}}
QPushButton#WinClose:hover {{
    background: {RED};
}}
QWidget#Sidebar {{
    background: rgba(255, 255, 255, 0.86);
    border-right: 1px solid {SEPARATOR};
    border-bottom-left-radius: 21px;
}}
QFrame#Card {{
    background: {CARD};
    border: 1px solid rgba(60, 60, 67, 0.08);
    border-radius: 18px;
}}
QFrame#InnerCard {{
    background: {CARD_ALT};
    border: 1px solid rgba(60, 60, 67, 0.06);
    border-radius: 14px;
}}
QFrame#HeroCard {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {BLUE}, stop:1 #5E5CE6);
    border: none;
    border-radius: 20px;
}}
QFrame#Divider {{
    background: {SEPARATOR};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}
QLabel {{
    background: transparent;
}}
QLabel#PageTitle {{
    font-size: 30px;
    font-weight: 700;
    color: {INK};
}}
QLabel#PageSubtitle {{
    font-size: 14px;
    color: {GRAY};
}}
QLabel#SectionTitle {{
    font-size: 17px;
    font-weight: 600;
    color: {INK};
}}
QLabel#CardTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {INK};
}}
QLabel#Caption, QLabel#Hint {{
    font-size: 12px;
    color: {GRAY};
}}
QLabel#HeroTitle {{
    color: white;
    font-size: 22px;
    font-weight: 700;
}}
QLabel#HeroSub {{
    color: rgba(255, 255, 255, 0.82);
    font-size: 13px;
}}
QLabel#MetricValue {{
    font-size: 28px;
    font-weight: 700;
    color: {INK};
}}
QLabel#MetricTitle {{
    font-size: 12px;
    color: {GRAY};
}}
QLabel#Badge {{
    background: {BLUE_SOFT};
    color: {BLUE_DARK};
    border-radius: 9px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 600;
}}
QLabel#BadgeGreen {{
    background: {GREEN_SOFT};
    color: #1B7F3B;
    border-radius: 9px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 600;
}}
QLabel#BadgeOrange {{
    background: {ORANGE_SOFT};
    color: #B35C00;
    border-radius: 9px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 600;
}}
QLabel#BadgeGray {{
    background: {FILL};
    color: {GRAY};
    border-radius: 9px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 600;
}}
QLabel#Formula {{
    background: {CARD_ALT};
    border: 1px solid {SEPARATOR};
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 15px;
    color: {INK};
}}
QLabel#EmptyTitle {{
    font-size: 16px;
    font-weight: 600;
    color: {INK_2};
}}
QLabel#EmptyText {{
    font-size: 13px;
    color: {GRAY};
}}
QPushButton {{
    border: none;
    border-radius: 11px;
    padding: 9px 16px;
    font-size: 13px;
    font-weight: 600;
    background: {FILL};
    color: {INK};
}}
QPushButton:hover {{
    background: {FILL_2};
}}
QPushButton:pressed {{
    background: #DEDEE5;
}}
QPushButton:disabled {{
    color: {GRAY_2};
    background: #F7F7FA;
}}
QPushButton[variant="primary"] {{
    background: {BLUE};
    color: white;
    padding: 10px 20px;
}}
QPushButton[variant="primary"]:hover {{
    background: #1A88FF;
}}
QPushButton[variant="primary"]:pressed {{
    background: {BLUE_DARK};
}}
QPushButton[variant="primary"]:disabled {{
    background: #B8D8FF;
    color: white;
}}
QPushButton[variant="secondary"] {{
    background: white;
    color: {BLUE};
    border: 1px solid rgba(0, 122, 255, 0.22);
}}
QPushButton[variant="secondary"]:hover {{
    background: {BLUE_SOFT};
}}
QPushButton[variant="secondary"]:disabled {{
    background: white;
    color: {GRAY_2};
    border: 1px solid {SEPARATOR};
}}
QPushButton[variant="ghost"]:disabled {{
    color: {GRAY_2};
    background: transparent;
}}
QPushButton[variant="chip"]:disabled {{
    color: {GRAY_2};
    background: {CARD_ALT};
}}
QPushButton[variant="ghost"] {{
    background: transparent;
    color: {BLUE};
    padding: 6px 10px;
}}
QPushButton[variant="ghost"]:hover {{
    background: {BLUE_SOFT};
}}
QPushButton[variant="danger"] {{
    background: {RED};
    color: white;
    padding: 10px 20px;
}}
QPushButton[variant="danger"]:hover {{
    background: #FF544A;
}}
QPushButton[variant="chip"] {{
    background: {FILL};
    color: {INK_2};
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
}}
QPushButton[variant="chip"]:hover {{
    background: {FILL_2};
}}
QPushButton[variant="chip"]:checked {{
    background: {BLUE_SOFT};
    color: {BLUE_DARK};
    border: 1px solid rgba(0, 122, 255, 0.30);
}}
QPushButton[variant="navItem"] {{
    background: transparent;
    color: {GRAY};
    text-align: left;
    padding: 0 12px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 500;
}}
QPushButton[variant="navItem"]:hover {{
    background: rgba(118, 118, 128, 0.08);
    color: {INK};
}}
QPushButton[variant="navItem"]:checked {{
    background: {BLUE_SOFT};
    color: {BLUE_DARK};
    font-weight: 600;
}}
QPushButton[variant="link"] {{
    background: transparent;
    color: {BLUE};
    padding: 2px 4px;
    font-size: 12px;
}}
QPushButton[variant="link"]:hover {{
    color: {BLUE_DARK};
    text-decoration: underline;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {FILL};
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 8px 10px;
    min-height: 22px;
    color: {INK};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    background: white;
    border: 1px solid {BLUE};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {GRAY_2};
    background: #F7F7FA;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background: white;
    border: 1px solid {SEPARATOR};
    border-radius: 10px;
    padding: 4px;
    selection-background-color: {BLUE_SOFT};
    selection-color: {INK};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 16px;
    border: none;
    background: transparent;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {FILL_2};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {BLUE};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 20px;
    height: 20px;
    margin: -8px 0;
    border-radius: 10px;
    background: white;
    border: 1px solid rgba(0, 0, 0, 0.10);
}}
QCheckBox {{
    spacing: 8px;
    color: {INK};
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid {GRAY_2};
    background: white;
}}
QCheckBox::indicator:checked {{
    background: {BLUE};
    border: 1px solid {BLUE};
}}
QProgressBar {{
    background: {FILL_2};
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {BLUE}, stop:1 #5E5CE6);
    border-radius: 5px;
}}
QTableWidget {{
    background: transparent;
    border: none;
    gridline-color: transparent;
    alternate-background-color: {CARD_ALT};
    font-size: 12px;
    selection-background-color: {BLUE_SOFT};
    selection-color: {INK};
}}
QTableWidget::item {{
    padding: 5px 8px;
    border: none;
}}
QHeaderView::section {{
    background: {FILL};
    color: {GRAY};
    border: none;
    padding: 7px 8px;
    font-size: 11px;
    font-weight: 600;
}}
QTableCornerButton::section {{
    background: {FILL};
    border: none;
}}
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
    font-size: 13px;
}}
QListWidget::item {{
    padding: 6px 8px;
    border-radius: 8px;
    margin: 1px 2px;
}}
QListWidget::item:hover {{
    background: {FILL};
}}
QListWidget::item:selected {{
    background: {BLUE_SOFT};
    color: {BLUE_DARK};
}}
QPlainTextEdit#Console {{
    background: #141417;
    color: #E7E7EC;
    border: none;
    border-radius: 14px;
    padding: 12px;
    font-size: 11px;
    selection-background-color: #264F78;
}}
QTabWidget::pane {{
    border: none;
    background: transparent;
}}
QTabBar::tab {{
    background: transparent;
    color: {GRAY};
    padding: 8px 16px;
    border-radius: 9px;
    font-weight: 600;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    background: white;
    color: {INK};
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: rgba(60, 60, 67, 0.22);
    min-height: 36px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical:hover {{
    background: rgba(60, 60, 67, 0.38);
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
    border: none;
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px 4px;
}}
QScrollBar::handle:horizontal {{
    background: rgba(60, 60, 67, 0.22);
    min-width: 36px;
    border-radius: 3px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
    border: none;
    width: 0;
}}
QSplitter::handle {{
    background: transparent;
    width: 12px;
}}
QToolTip {{
    background: {INK};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12px;
}}
"""
