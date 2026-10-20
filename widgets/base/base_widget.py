"""Common lifecycle for reusable Qt widgets."""

from abc import abstractmethod

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


class BaseWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setup_ui()
        self.connect_signals()
        self.apply_style()

    @abstractmethod
    def setup_ui(self) -> None:
        """Build child widgets and layouts."""
        raise NotImplementedError

    def connect_signals(self) -> None:
        """Optional hook for widgets with interactive controls."""

    def apply_style(self) -> None:
        """Optional hook for widget-specific styling."""
