#!/usr/bin/env python3

import sys
import time
import configparser
import enum
import logging
from typing import Optional

# UI Imports
from PyQt6.QtCore import (Qt, QTimer, pyqtSignal, QObject, QThread)
from PyQt6.QtGui import QPixmap, QImage, QCursor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QCheckBox, QFrame, QFormLayout, QHBoxLayout,
                             QVBoxLayout, QSpinBox, QPushButton,
                             QMessageBox, QProgressBar, QListWidget,
                             QInputDialog, QDialog)

# Logic Imports
import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab, ImageQt
from pynput.keyboard import Listener, KeyCode

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

__version__ = '0.9'
__author__ = 'Gemini'


# --- WORKER THREAD ---

class FisherWorker(QThread):
    """
    Runs image processing in background.
    """
    image_processed = pyqtSignal(QImage, QImage)
    stats_updated = pyqtSignal(float, str)
    potion_drank = pyqtSignal(int)

    def __init__(self, config_data):
        super().__init__()
        self.running = True
        self.paused = False
        self.cfg = config_data
        self.tracker = MovementTracker()
        self.logic = FisherLogic()
        self.potion_timer = time.time()

    def update_config(self, new_config):
        self.cfg = new_config

    def run(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            cx, cy = self.cfg['x'], self.cfg['y']
            shift = 50
            bbox = (cx - shift, cy - shift, cx + shift, cy + shift)

            try:
                pil_img = ImageGrab.grab(bbox=bbox)
            except OSError:
                time.sleep(0.1);
                continue

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            diff_img = self.tracker.get_diff(gray, self.cfg['threshold'])

            count = cv2.countNonZero(diff_img)
            area = (shift * 2) ** 2
            sense_val = count * self.cfg['sensitivity'] / area

            state_desc = self.logic.update(sense_val)

            # Potion logic
            if self.cfg['use_potions']:
                time_left = int((self.potion_timer + self.cfg['potion_delay']) - time.time())
                if time_left <= 0:
                    # We pass the Virtual Key Code to press_key
                    MouseController.press_vk(self.cfg['potion_key_vk'])
                    self.potion_timer = time.time()
                    self.potion_drank.emit(0)
                else:
                    self.potion_drank.emit(time_left)
            else:
                self.potion_drank.emit(-1)

            q_raw = ImageQt.ImageQt(pil_img)
            h, w = diff_img.shape
            q_proc = QImage(diff_img.data, w, h, w, QImage.Format.Format_Grayscale8)

            self.image_processed.emit(q_raw, q_proc)
            self.stats_updated.emit(sense_val, state_desc)
            time.sleep(0.05)

    def stop(self):
        self.running = False
        self.wait()


# --- INPUT & LOGIC ---

class KeyMonitor(QObject):
    """
    Listens for hardware Key Codes (VK) instead of Chars.
    This fixes the RU/EN layout issue.
    """
    # Signal now emits the Integer Virtual Key Code (e.g., 70 for 'F')
    keyPressed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        try:
            vk = None
            # Try to get the hardware code
            if hasattr(key, 'vk') and key.vk is not None:
                vk = key.vk
            # Fallback for some special keys (though VK usually exists on Windows)
            elif hasattr(key, 'value') and key.value:
                vk = key.value.vk

            if vk:
                self.keyPressed.emit(vk)
        except Exception:
            pass


class MouseController:
    @staticmethod
    def click():
        pyautogui.mouseDown()
        time.sleep(0.05)
        pyautogui.mouseUp()

    @staticmethod
    def press_vk(vk_code):
        """Press key by Virtual Key Code (Layout independent)"""
        # PyAutoGUI works better with chars, but pynput works with codes.
        # We need to convert VK back to a char for PyAutoGUI,
        # OR use pynput Controller for output too.
        # Here we use a safe fallback:
        try:
            # Attempt to convert VK to char just for PyAutoGUI
            char = KeyCode.from_vk(vk_code).char
            if char:
                pyautogui.press(char)
            else:
                # If it's a special key, we might ignore it or add logic
                pass
        except:
            pass


class FisherState(enum.Enum):
    INIT = "Waiting for user..."
    CAST = "Casting the line"
    WAIT = "Waiting for movement"
    REEL = "Hooked - reeling in"


class FisherLogic:
    def __init__(self):
        self.state = FisherState.INIT
        self.last_action = time.time()

    def update(self, sense_level: float) -> str:
        now = time.time()
        if self.state == FisherState.INIT:
            if sense_level > 1:
                self._switch(FisherState.CAST)
                MouseController.click()
        elif self.state == FisherState.CAST:
            if (now - self.last_action) > 1.5 and sense_level < 1:
                self._switch(FisherState.WAIT)
        elif self.state == FisherState.WAIT:
            if (now - self.last_action) > 1.0 and sense_level > 1:
                self._switch(FisherState.REEL)
                MouseController.click()
        elif self.state == FisherState.REEL:
            if (now - self.last_action) > 2.0 and sense_level < 1:
                self._switch(FisherState.CAST)
                MouseController.click()
        return self.state.value

    def _switch(self, new_state):
        self.state = new_state
        self.last_action = time.time()


class MovementTracker:
    def __init__(self, size=3):
        self.buffer = []
        self.size = size

    def get_diff(self, img, threshold):
        self.buffer.append(img)
        if len(self.buffer) > self.size: self.buffer.pop(0)
        if len(self.buffer) < 3: return np.zeros_like(img)
        t0, t1, t2 = self.buffer[-3:]
        d1 = cv2.absdiff(t2, t1);
        d2 = cv2.absdiff(t1, t0)
        res = cv2.bitwise_or(d1, d2)
        _, res = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY)
        return res


# --- MAIN GUI ---

class AppUi(QMainWindow):
    SHIFT = 50
    STYLE_GREEN = "QPushButton { background-color: #4caf50; color: black; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #388e3c; }"
    STYLE_RED = "QPushButton { background-color: #d32f2f; color: white; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #b71c1c; }"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'AutoFisher v{__version__}')

        self.config = configparser.ConfigParser()
        self.worker: Optional[FisherWorker] = None
        self.hotkeys_active = True

        # We now store VK codes (Integers), e.g., 70 for 'F'
        self.vk_fishing = 70  # Default F
        self.vk_pos = 86  # Default V

        # We also store text for buttons
        self.txt_fishing = 'F'
        self.txt_pos = 'V'

        self._init_ui()
        self._load_config()

        self.key_monitor = KeyMonitor()
        self.key_monitor.keyPressed.connect(self._handle_hotkey_vk)

    def _init_ui(self):
        central = QWidget();
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        left = QVBoxLayout();
        right = QVBoxLayout()
        main_layout.addLayout(left);
        main_layout.addLayout(right)

        # Hotkey Toggle
        self.btn_hotkeys = QPushButton()
        self.btn_hotkeys.setFixedHeight(40)
        self.btn_hotkeys.clicked.connect(self._toggle_hotkeys_state)
        self._refresh_hotkey_btn()
        left.addWidget(self.btn_hotkeys)

        # Previews
        prev_layout = QHBoxLayout()
        self.lbl_raw = self._create_preview("Raw");
        self.lbl_proc = self._create_preview("Mask")
        prev_layout.addWidget(self.lbl_raw);
        prev_layout.addWidget(self.lbl_proc)
        left.addLayout(prev_layout)

        # Status
        self.prog_bar = QProgressBar();
        self.lbl_state = QLabel("State: IDLE")
        self.lbl_potion = QLabel("Potions: OFF");
        self.lbl_mouse = QLabel("Mouse: N/A")
        left.addWidget(self.prog_bar);
        left.addWidget(self.lbl_mouse)
        left.addWidget(self.lbl_state);
        left.addWidget(self.lbl_potion)

        # Form
        form = QFormLayout()
        self.spin_x = QSpinBox();
        self.spin_x.setRange(0, 5000)
        self.spin_y = QSpinBox();
        self.spin_y.setRange(0, 5000)

        self.btn_pos_key = QPushButton(f"Set Pos Key ({self.txt_pos})")
        self.btn_pos_key.clicked.connect(lambda: self._wait_key('pos'))

        self.spin_th = QSpinBox();
        self.spin_th.setRange(0, 255);
        self.spin_th.setValue(10)
        self.spin_sens = QSpinBox();
        self.spin_sens.setRange(0, 2000);
        self.spin_sens.setValue(50)
        self.chk_pot = QCheckBox()
        self.spin_delay = QSpinBox();
        self.spin_delay.setRange(0, 3600)

        for w in [self.spin_x, self.spin_y, self.spin_th, self.spin_sens, self.chk_pot, self.spin_delay]:
            if isinstance(w, QSpinBox):
                w.valueChanged.connect(self._update_worker_config)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self._update_worker_config)

        form.addRow("X:", self.spin_x);
        form.addRow("Y:", self.spin_y)
        form.addRow("Pos Hotkey:", self.btn_pos_key)
        form.addRow("Threshold:", self.spin_th);
        form.addRow("Sensitivity:", self.spin_sens)
        form.addRow("Potions:", self.chk_pot);
        form.addRow("Delay:", self.spin_delay)
        left.addLayout(form)

        # Controls
        self.btn_start = QPushButton("Start Fishing")
        self.btn_start.setFixedHeight(30);
        self.btn_start.setStyleSheet(self.STYLE_GREEN)
        self.btn_start.clicked.connect(self._toggle_worker)
        left.addWidget(self.btn_start)

        self.btn_main_key = QPushButton(f"Change Start Key ({self.txt_fishing})")
        self.btn_main_key.clicked.connect(lambda: self._wait_key('main'))
        left.addWidget(self.btn_main_key)

        btn_save = QPushButton("Save Preset");
        btn_save.clicked.connect(self._save_preset)
        left.addWidget(btn_save)

        # Presets
        right.addWidget(QLabel("Presets:"))
        self.list_presets = QListWidget();
        self.list_presets.itemClicked.connect(self._load_preset_values)
        right.addWidget(self.list_presets)
        b_add = QPushButton("New");
        b_add.clicked.connect(self._add_preset)
        b_del = QPushButton("Delete");
        b_del.clicked.connect(self._del_preset)
        right.addWidget(b_add);
        right.addWidget(b_del)

    # --- HELPERS ---
    def _create_preview(self, t):
        l = QLabel();
        l.setFixedSize(100, 100);
        l.setFrameShape(QFrame.Shape.Box);
        l.setToolTip(t)
        return l

    def _get_ui_config(self):
        # We need to find the VK for the potion key.
        # For simplicity, let's assume potion key is standard 'B' (VK 66) unless saved otherwise.
        # Ideally, you'd add a UI button to set potion key too.
        return {
            'x': self.spin_x.value(), 'y': self.spin_y.value(),
            'threshold': self.spin_th.value(), 'sensitivity': self.spin_sens.value(),
            'use_potions': self.chk_pot.isChecked(), 'potion_delay': self.spin_delay.value(),
            'potion_key_vk': 66  # Default 'B'
        }

    # --- WORKER ---
    def _toggle_worker(self):
        if self.worker:
            self.worker.stop();
            self.worker = None
            self.btn_start.setText("Start Fishing");
            self.btn_start.setStyleSheet(self.STYLE_GREEN)
            self.lbl_state.setText("State: STOPPED");
            self._set_inputs_enabled(True)
        else:
            self.worker = FisherWorker(self._get_ui_config())
            self.worker.image_processed.connect(self._on_worker_image)
            self.worker.stats_updated.connect(self._on_worker_stats)
            self.worker.potion_drank.connect(self._on_worker_potion)
            self.worker.start()
            self.btn_start.setText("Stop Fishing");
            self.btn_start.setStyleSheet(self.STYLE_RED)
            self._set_inputs_enabled(False)

    def _update_worker_config(self):
        if self.worker: self.worker.update_config(self._get_ui_config())
        if not self.worker:
            pos = QCursor.pos();
            self.lbl_mouse.setText(f"Mouse: {pos.x()}, {pos.y()}")

    def _on_worker_image(self, raw, proc):
        self.lbl_raw.setPixmap(QPixmap.fromImage(raw));
        self.lbl_proc.setPixmap(QPixmap.fromImage(proc))

    def _on_worker_stats(self, s, t):
        self.lbl_state.setText(f"State: {t}");
        self.prog_bar.setValue(min(100, int(s * 100)))

    def _on_worker_potion(self, t):
        self.lbl_potion.setText(f"Drink in: {t}s" if t > 0 else ("DRANK!" if t == 0 else "OFF"))

    # --- NEW HOTKEY LOGIC (VK BASED) ---
    def _toggle_hotkeys_state(self):
        self.hotkeys_active = not self.hotkeys_active
        self._refresh_hotkey_btn()

    def _refresh_hotkey_btn(self):
        if self.hotkeys_active:
            self.btn_hotkeys.setText("HOTKEYS: LISTENING (Global)");
            self.btn_hotkeys.setStyleSheet(self.STYLE_RED)
        else:
            self.btn_hotkeys.setText("HOTKEYS: IGNORED");
            self.btn_hotkeys.setStyleSheet(self.STYLE_GREEN)

    def _handle_hotkey_vk(self, pressed_vk):
        """Compares integers (Virtual Key Codes)"""
        if not self.hotkeys_active: return

        # print(f"Debug: Pressed VK {pressed_vk}") # Uncomment to debug

        if pressed_vk == self.vk_fishing:
            self._toggle_worker()
        elif pressed_vk == self.vk_pos:
            pos = pyautogui.position()
            self.spin_x.setValue(pos.x);
            self.spin_y.setValue(pos.y)

    def _wait_key(self, target):
        d = QDialog(self);
        d.setWindowTitle("Press Key");
        d.setFixedSize(200, 100)
        l = QLabel("Press any key...", d);
        l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(d);
        layout.addWidget(l)

        # We need a temporary listener just for this dialog
        def on_press(key):
            vk = None
            char_display = "Unknown"

            if hasattr(key, 'vk') and key.vk:
                vk = key.vk
                # Try to get readable name
                if hasattr(key, 'char') and key.char:
                    char_display = key.char.upper()
                else:
                    char_display = f"VK_{vk}"
            elif hasattr(key, 'value') and key.value:
                vk = key.value.vk
                char_display = key.name.upper()

            if vk:
                if target == 'main':
                    self.vk_fishing = vk
                    self.txt_fishing = char_display
                    self.btn_main_key.setText(f"Start Key ({char_display})")
                elif target == 'pos':
                    self.vk_pos = vk
                    self.txt_pos = char_display
                    self.btn_pos_key.setText(f"Pos Key ({char_display})")

                # Stop listener and close dialog safely
                d.accept()
                return False  # Stop listener

        listener = Listener(on_press=on_press)
        listener.start()
        d.exec()
        listener.stop()  # Ensure stopped if dialog closed manually

    # --- CONFIG ---
    def _set_inputs_enabled(self, val):
        self.list_presets.setEnabled(val)

    def _load_config(self):
        self.config.read('config.ini')
        if not self.config.sections(): self.config['DEFAULT'] = {'screen_x': '800', 'screen_y': '600'}
        self._refresh_list()

    def _refresh_list(self):
        self.list_presets.clear();
        self.list_presets.addItem("DEFAULT")
        for s in self.config.sections(): self.list_presets.addItem(s)
        self.list_presets.setCurrentRow(0)

    def _get_preset_name(self):
        i = self.list_presets.currentItem();
        return i.text() if i else 'DEFAULT'

    def _load_preset_values(self):
        name = self._get_preset_name()
        if name == "DEFAULT" or name in self.config:
            d = self.config[name]
            self.spin_x.setValue(int(d.get('screen_x', 800)))
            self.spin_y.setValue(int(d.get('screen_y', 600)))
            self.spin_th.setValue(int(d.get('threshold', 10)))
            self.spin_sens.setValue(int(d.get('sensitivity', 50)))
            # Load Hotkeys VKs if saved (You might want to save them too)

    def _save_preset(self):
        name = self._get_preset_name()
        if name not in self.config: self.config[name] = {}
        self.config[name]['screen_x'] = str(self.spin_x.value())
        self.config[name]['screen_y'] = str(self.spin_y.value())
        self.config[name]['threshold'] = str(self.spin_th.value())
        self.config[name]['sensitivity'] = str(self.spin_sens.value())
        with open('config.ini', 'w') as f: self.config.write(f)
        QMessageBox.information(self, "Saved", f"Saved to {name}")

    def _add_preset(self):
        t, o = QInputDialog.getText(self, 'New', 'Name:')
        if o and t:
            self.config[t] = {};
            self.list_presets.addItem(t)
            self.list_presets.setCurrentRow(self.list_presets.count() - 1);
            self._save_preset()

    def _del_preset(self):
        n = self._get_preset_name()
        if n == 'DEFAULT': return
        self.config.remove_section(n);
        self._refresh_list()
        with open('config.ini', 'w') as f: self.config.write(f)


def main():
    app = QApplication(sys.argv)
    window = AppUi()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()