"""Shared UI building blocks: style constants, safe coercion helpers,
default hotkey VK codes, and VK → display-char conversion.
"""

import logging

from pynput.keyboard import KeyCode


# --- button / progress-bar style strings ---
STYLE_GREEN = "QPushButton { background-color: #4caf50; color: black; font-weight: bold; border-radius: 4px; }"
STYLE_RED = "QPushButton { background-color: #d32f2f; color: white; font-weight: bold; border-radius: 4px; }"
STYLE_ORANGE = "QPushButton { background-color: #ff9800; color: black; font-weight: bold; border-radius: 4px; }"
STYLE_LISTENING = "background-color: #f44336; color: white; font-weight: bold;"
STYLE_BAR_BLUE = "QProgressBar::chunk { background-color: #2196F3; }"
STYLE_BAR_GREEN = "QProgressBar::chunk { background-color: #4CAF50; }"

# --- default hotkey VK codes ---
DEFAULT_VK_FISHING = 70  # F
DEFAULT_VK_POS = 86      # V
DEFAULT_VK_TRAIN = 84    # T
DEFAULT_VK_POTION = 66   # B


def safe_int(d, key: str, default: int) -> int:
    try:
        return int(d.get(key, default))
    except (ValueError, TypeError):
        logging.warning(f"Profile field '{key}'={d.get(key)!r} is not an int; using {default}")
        return default


def safe_float(d, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except (ValueError, TypeError):
        logging.warning(f"Profile field '{key}'={d.get(key)!r} is not a float; using {default}")
        return default


def vk_to_char(vk: int) -> str:
    """Best-effort VK code → display label (e.g. 70 → 'F', 112 → 'VK_112')."""
    try:
        ch = KeyCode.from_vk(vk).char
        return ch.upper() if ch else f"VK_{vk}"
    except Exception:
        return f"VK_{vk}"
