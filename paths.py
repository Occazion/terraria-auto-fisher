"""Filesystem path helpers — bundle resources, user data dir, pattern cleanup."""

import glob
import os
import sys


def resource_path(relative_path: str) -> str:
    """Resolve a bundled asset path.

    Inside a PyInstaller bundle, assets live under sys._MEIPASS. Running from
    source, they live next to this file (which is at the project root).
    """
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def get_app_data_dir() -> str:
    """Return %APPDATA%/AutoFisher/Patterns, creating it if missing."""
    app_data = os.getenv('APPDATA') or os.path.expanduser('~')
    path = os.path.join(app_data, 'AutoFisher', 'Patterns')
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_temp_patterns() -> None:
    """Delete every Splash_*.wav under the patterns dir.

    Called on app startup so the next session starts fresh. The bundled
    Splash_1.wav lives next to the .exe (or in MEIPASS) and is untouched.
    """
    folder = get_app_data_dir()
    for f in glob.glob(os.path.join(folder, "Splash_*.wav")):
        try:
            os.remove(f)
        except OSError:
            pass
