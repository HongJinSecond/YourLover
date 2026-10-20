"""Shared colors, typography and controls for the desktop surfaces."""

from pathlib import Path

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QColor, QFont, QGuiApplication
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget

BACKGROUND = "#F7F8FC"
SURFACE = "#FFFFFF"
SIDEBAR = "#F0F2F8"
TEXT = "#252A3B"
MUTED = "#7B8195"
BORDER = "#E5E7F0"
ACCENT = "#7762C5"
ACCENT_HOVER = "#6651B4"
ACCENT_SOFT = "#EDE9F9"
SUCCESS = "#47856E"
_CHEVRON = (Path(__file__).resolve().parents[1] / "assets" / "ui" / "chevron_down.svg").as_posix()


def ui_font(size: int = 10, bold: bool = False) -> QFont:
    font = QFont()
    font.setFamilies(["Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", "sans-serif"])
    font.setPointSize(size)
    font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
    return font


SCROLL_STYLE = f"""
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 6px; margin: 4px 0; }}
QScrollBar::handle:vertical {{ background: #D5D7E3; border-radius: 3px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
"""

BUTTON_STYLE = f"""
QPushButton {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 9px 16px; font-weight: 500;
}}
QPushButton:hover {{ background: {BACKGROUND}; border-color: #D0CCDF; }}
QPushButton:pressed {{ background: {ACCENT_SOFT}; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #A0A5B5; background: #F1F2F6; border-color: #EBEDF4; }}
QPushButton[primary="true"] {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[primary="true"]:disabled {{ background: #D2CAEB; border-color: #D2CAEB; }}
QPushButton[iconOnly="true"] {{ padding: 0; background: transparent; border-color: transparent; }}
QPushButton[iconOnly="true"]:hover {{ background: {ACCENT_SOFT}; }}
"""

FIELD_STYLE = f"""
QLineEdit, QComboBox, QPlainTextEdit {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 11px 12px; selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
}}
QLineEdit:hover, QComboBox:hover, QPlainTextEdit:hover {{ border-color: #C9C5DA; }}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {ACCENT}; background: #FDFCFF; }}
QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled {{ color: {MUTED}; background: {BACKGROUND}; }}
QComboBox::drop-down {{ width: 30px; border: none; }}
QComboBox::down-arrow {{ image: url("{_CHEVRON}"); width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{
    color: {TEXT}; background: {SURFACE}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT}; padding: 6px;
}}
"""


def add_shadow(widget: QWidget, blur: int = 24, offset: int = 5) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, offset)
    effect.setColor(QColor(37, 42, 59, 24))
    widget.setGraphicsEffect(effect)


def screen_bounds(anchor: QWidget | None = None) -> QRect:
    screen = anchor.screen() if anchor is not None else QGuiApplication.primaryScreen()
    return screen.availableGeometry() if screen is not None else QRect(0, 0, 1920, 1080)


def keep_on_screen(widget: QWidget, point: QPoint, anchor: QWidget | None = None) -> QPoint:
    bounds = screen_bounds(anchor).adjusted(8, 8, -8, -8)
    x = min(max(point.x(), bounds.left()), max(bounds.left(), bounds.right() - widget.width() + 1))
    y = min(max(point.y(), bounds.top()), max(bounds.top(), bounds.bottom() - widget.height() + 1))
    return QPoint(x, y)
