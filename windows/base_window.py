"""Shared window lifecycle: close secondary surfaces without destroying them."""

from abc import abstractmethod

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget


class BaseWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_window_flags()
        self.setup_ui()
        self.connect_signals()
        self.apply_style()

    def _setup_window_flags(self) -> None:
        self.setWindowFlags(Qt.WindowType.Window)

    @abstractmethod
    def setup_ui(self) -> None:
        raise NotImplementedError

    def connect_signals(self) -> None:
        """Optional hook for a window's controls and event subscriptions."""

    def apply_style(self) -> None:
        """Optional hook for window-specific styles."""

    def show_window(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def hide_window(self) -> None:
        self.hide()

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide_window()
