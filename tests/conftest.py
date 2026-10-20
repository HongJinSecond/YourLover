"""Keep Qt and import-time RAG initialization isolated from the user's desktop data."""

import inspect
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from utils.config_manager import config

_runtime = Path(tempfile.mkdtemp(prefix="yourlover-pytest-"))
config.update(
    {"CHROMA_PATH": str(_runtime / "chroma"), "MD5_PATH": str(_runtime / "md5.txt")}, save=False
)


@pytest.fixture(autouse=True)
def isolate_default_memory_stores(tmp_path, monkeypatch):
    from agent.affinity.affinity_store import AffinityStore
    from agent.memory.profile_store import ProfileStore
    from agent.memory.session_store import SessionStore
    from agent.memory.store import FileMemoryStore

    def replace_default(cls, parameter, destination):
        original = cls.__init__
        signature = inspect.signature(original)

        def initialize(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            if bound.arguments.get(parameter) is None:
                bound.arguments[parameter] = destination
            original(*bound.args, **bound.kwargs)

        monkeypatch.setattr(cls, "__init__", initialize)

    for cls, name in (
        (AffinityStore, "affinity"),
        (ProfileStore, "profiles"),
        (SessionStore, "sessions"),
    ):
        replace_default(cls, "database_path", tmp_path / f"{name}.sqlite3")
    replace_default(FileMemoryStore, "root_dir", tmp_path / "users")


@pytest.fixture(scope="session")
def qt_app():
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
