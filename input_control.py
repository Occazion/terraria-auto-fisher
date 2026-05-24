"""Mouse/keyboard input + global hotkey listener.

Wraps pyautogui for mouse clicks and pynput for keyboard simulation and
global key event capture. KeyMonitor lives here so the rest of the app
only has to import one module for "user input control."
"""

import logging
import time

import pyautogui
from pynput.keyboard import Listener, KeyCode, Controller as KeyboardController
from PyQt6.QtCore import QObject, pyqtSignal


# A single shared keyboard Controller — creating one is cheap but they hold a
# native handle, so reuse rather than re-create on every keypress.
_keyboard_ctrl = KeyboardController()

# Default click hold time. Short enough to register as a tap, long enough that
# Terraria's input layer doesn't drop it.
_CLICK_HOLD_SEC = 0.05
_KEY_HOLD_SEC = 0.03


class MouseController:
    """Stateless wrapper around pyautogui mouse + a robust VK key press."""

    @staticmethod
    def click() -> None:
        pyautogui.mouseDown()
        time.sleep(_CLICK_HOLD_SEC)
        pyautogui.mouseUp()

    @staticmethod
    def press_vk(vk_code) -> None:
        """Press the key identified by Windows virtual-key code.

        Uses the pynput Controller directly so any VK works — including keys
        whose KeyCode.char is None (function keys, numpad, etc.).
        """
        try:
            key = KeyCode.from_vk(int(vk_code))
            _keyboard_ctrl.press(key)
            time.sleep(_KEY_HOLD_SEC)
            _keyboard_ctrl.release(key)
        except Exception as e:
            logging.warning(f"press_vk({vk_code}) failed: {e}")


def tick_potion(cfg, potion_timer):
    """Advance the potion timer by one tick. Returns (new_timer, signal_value).

    Signal value semantics (matches the UI label expectations):
        -1 → potions disabled
         0 → just pressed the key this tick
        >0 → seconds remaining until next press
    """
    if not cfg.get('use_potions'):
        return potion_timer, -1
    delay = int(cfg['potion_delay'])
    time_left = int((potion_timer + delay) - time.time())
    if time_left <= 0:
        MouseController.press_vk(cfg.get('potion_vk', 66))
        return time.time(), 0
    return potion_timer, time_left


class KeyMonitor(QObject):
    """Global keyboard listener that emits a Qt signal on every keypress.

    Runs the pynput Listener on its own OS thread; the signal is delivered to
    Qt slots via a queued connection, so receivers run on the GUI thread.
    """

    keyPressed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key) -> None:
        try:
            vk = key.vk if hasattr(key, 'vk') else key.value.vk
            if vk:
                self.keyPressed.emit(vk)
        except Exception:
            pass

    def stop(self) -> None:
        self.listener.stop()
