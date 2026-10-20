"""Small vector icons rendered locally; independent of emoji fonts."""

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from widgets.theme import MUTED

_PATHS = {
    "chat": '<path d="M20 11.5a8 8 0 0 1-8 8H5l-4 3v-11a9 9 0 1 1 19 0Z" transform="translate(1 -1) scale(.95)"/><path d="M7 9h8M7 13h5"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "send": '<path d="m21 3-7 18-4-7-7-4 18-7ZM10 14 21 3"/>',
    "close": '<path d="m6 6 12 12M18 6 6 18"/>',
    "minimize": '<path d="M5 12h14"/>',
    "maximize": '<rect x="5" y="5" width="14" height="14" rx="1"/>',
    "trash": '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
    "settings": '<path d="m9 3-.8 2.4-2.5 1-.2 2.4L3 10v4l2.5 1.2.2 2.4 2.5 1L9 21h6l.8-2.4 2.5-1 .2-2.4L21 14v-4l-2.5-1.2-.2-2.4-2.5-1L15 3Z"/><circle cx="12" cy="12" r="3"/>',
    "eye": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
    "eye_off": '<path d="m3 3 18 18M10 5a11 11 0 0 1 12 7 18 18 0 0 1-3 4M6 6a18 18 0 0 0-4 6s4 7 10 7a11 11 0 0 0 5-1M10 10a3 3 0 0 0 4 4"/>',
    "spark": '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z"/>',
}


@lru_cache(maxsize=64)
def icon(name: str, color: str = MUTED, size: int = 20) -> QIcon:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round">{_PATHS[name]}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    result = QIcon()
    for scale in (1, 2):
        pixmap = QPixmap(size * scale, size * scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter, QRectF(0, 0, size * scale, size * scale))
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        result.addPixmap(pixmap)
    return result
