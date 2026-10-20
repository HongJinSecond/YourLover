"""Load, preview and import local chat avatars without depending on the working directory."""

import hashlib
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSaveFile, QSize, Qt
from PySide6.QtGui import QImage, QImageReader, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QLabel


def avatar_path(value: str, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_avatar(path: Path | None) -> QImage:
    if path is None or not path.is_file():
        return QImage()
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > 512:
        reader.setScaledSize(size.scaled(QSize(512, 512), Qt.AspectRatioMode.KeepAspectRatio))
    return reader.read()


def set_avatar(label: QLabel, path: Path | None) -> None:
    """Center-crop an image into the label; absent/invalid files retain the default Y."""
    image = read_avatar(path)
    label.clear()
    if image.isNull():
        label.setText("Y")
        return
    ratio = label.devicePixelRatioF()
    size = label.size()
    scaled = QPixmap.fromImage(image).scaled(
        round(size.width() * ratio),
        round(size.height() * ratio),
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    result = QPixmap(round(size.width() * ratio), round(size.height() * ratio))
    result.setDevicePixelRatio(ratio)
    result.fill(Qt.GlobalColor.transparent)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    clip = QPainterPath()
    clip.addRoundedRect(0, 0, size.width(), size.height(), 13, 13)
    painter.setClipPath(clip)
    scaled.setDevicePixelRatio(ratio)
    painter.drawPixmap(
        round((size.width() - scaled.width() / ratio) / 2),
        round((size.height() - scaled.height() / ratio) / 2),
        scaled,
    )
    painter.end()
    label.setPixmap(result)


def import_avatar(value: str, root: Path) -> str:
    """Save an uploaded image as a managed PNG; keep existing icon files in place."""
    path = avatar_path(value, root)
    if path is None:
        return ""
    image = read_avatar(path)
    if image.isNull():
        raise ValueError("无法读取头像，请选择有效的图片文件。")
    directory = root / "icon"
    if path.resolve().parent == directory.resolve():
        return path.resolve().relative_to(root.resolve()).as_posix()
    image = image.scaled(
        512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise ValueError("无法转换头像图片，请选择其他图片。")
    buffer.close()
    filename = f"avatar-{hashlib.sha256(bytes(data)).hexdigest()[:20]}.png"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    file = QSaveFile(str(destination))
    if not file.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError("无法保存头像，请检查 icon 目录的写入权限。")
    if file.write(data) != data.size() or not file.commit():
        raise OSError("头像保存失败，请重试。")
    return f"icon/{filename}"
