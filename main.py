#!/usr/bin/env python3

import sys
import time
import configparser
import enum
import logging
from typing import Optional

# UI Imports
from PyQt6.QtCore import (Qt, QTimer, pyqtSignal, QObject, QThread,
                          QSize, QPoint)
from PyQt6.QtGui import QPixmap, QImage, QCursor, QAction
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QCheckBox, QFrame, QFormLayout, QHBoxLayout,
                             QVBoxLayout, QSpinBox, QPushButton,
                             QMessageBox, QProgressBar, QListWidget,
                             QInputDialog, QDialog, QSystemTrayIcon)

# Logic Imports
import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab, ImageQt
from pynput.keyboard import Listener, KeyCode

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

__version__ = '0.5'
__author__ = 'Alexei Metlitski'


class KeyMonitor(QObject):
    """
    Мост между потоком pynput и PyQt.
    Обеспечивает потокобезопасность через сигналы.
    """
    keyPressed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.keyPressed.emit(key.char)
            else:
                # Обработка спецклавиш если нужно
                pass
        except Exception:
            pass

    def stop(self):
        self.listener.stop()


class MouseController:
    """Обертка для управления мышью."""

    @staticmethod
    def click():
        # PyAutoGUI автоматически обрабатывает задержки и платформы
        # Если нужен xdotool для Linux (античит), можно вернуть subprocess
        pyautogui.mouseDown()
        time.sleep(0.02)
        pyautogui.mouseUp()

    @staticmethod
    def press_key(key: str, duration: float = 0.02):
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)


class FisherState(enum.Enum):
    INIT = "Waiting for user..."
    CAST = "Casting the line"
    WAIT = "Waiting for movement"
    REEL = "Hooked - reeling in"


class FisherLogic:
    """Вся логика конечного автомата вынесена сюда."""

    def __init__(self):
        self.state = FisherState.INIT
        self.last_action_time = time.time()
        self.start_cast_time = 0

    def update(self, sense_level: float) -> FisherState:
        now = time.time()

        if self.state == FisherState.INIT:
            if sense_level > 1:
                self._switch_state(FisherState.CAST)
                MouseController.click()  # Cast start

        elif self.state == FisherState.CAST:
            # Ждем 1 секунду после заброса, пока поплавок успокоится
            if (now - self.last_action_time) > 1.0 and sense_level < 1:
                self._switch_state(FisherState.WAIT)

        elif self.state == FisherState.WAIT:
            # Если прошло больше 1 сек и есть движение -> клюнуло
            if (now - self.last_action_time) > 1.0 and sense_level > 1:
                self._switch_state(FisherState.REEL)
                MouseController.click()  # Подсекаем

        elif self.state == FisherState.REEL:
            # Ждем пока анимация закончится (0.5 сек)
            if (now - self.last_action_time) > 0.5 and sense_level < 1:
                self._switch_state(FisherState.CAST)
                MouseController.click()  # Новый заброс

        return self.state

    def _switch_state(self, new_state):
        self.state = new_state
        self.last_action_time = time.time()
        logging.info(f"State changed to: {new_state.name}")


class MovementTracker:
    def __init__(self, buffer_size=3):
        self.buffer = []
        self.size = buffer_size

    def get_diff(self, img_gray, threshold):
        """Возвращает маску изменений."""
        self.buffer.append(img_gray)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

        if len(self.buffer) < 3:
            return np.zeros_like(img_gray)

        t0, t1, t2 = self.buffer[-3:]
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        res = cv2.bitwise_or(d1, d2)
        _, res = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY)
        return res


class AppUi(QMainWindow):
    SHIFT = 50  # Размер зоны захвата (половина стороны)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'AutoFisher v{__version__}')
        self.config = configparser.ConfigParser()
        self.tracker = MovementTracker()
        self.fisher_logic = None  # None когда выключен

        # UI State vars
        self.hotkey_fishing = 'f'
        self.hotkey_pos = 'v'
        self.potion_timer = 0

        # Init components
        self._init_ui()
        self._load_initial_config()

        # Key Listener (Thread Safe)
        self.key_monitor = KeyMonitor()
        self.key_monitor.keyPressed.connect(self._handle_global_hotkey)

        # Main Loop Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._game_loop)
        self.timer.start(66)  # ~15 FPS

    def _init_ui(self):
        """Настройка интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Previews
        preview_layout = QHBoxLayout()
        self.lbl_raw = self._create_preview_label()
        self.lbl_proc = self._create_preview_label()
        preview_layout.addWidget(self.lbl_raw)
        preview_layout.addWidget(self.lbl_proc)
        left_layout.addLayout(preview_layout)

        # Status Widgets
        self.progress_bar = QProgressBar()
        self.lbl_status_mouse = QLabel("Mouse: N/A")
        self.lbl_status_state = QLabel("State: IDLE")
        self.lbl_status_potion = QLabel("Potions: OFF")

        for w in [self.progress_bar, self.lbl_status_mouse, self.lbl_status_state, self.lbl_status_potion]:
            left_layout.addWidget(w)

        # Controls Form
        form = QFormLayout()
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 5000)
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 5000)
        self.btn_set_pos_key = QPushButton(f"Set Pos Hotkey ({self.hotkey_pos})")
        self.btn_set_pos_key.clicked.connect(lambda: self._wait_for_hotkey('pos'))

        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_sens = QSpinBox()
        self.spin_sens.setRange(0, 2000)

        self.chk_potions = QCheckBox()
        self.spin_potion_delay = QSpinBox()
        self.spin_potion_delay.setRange(0, 3600)

        form.addRow("Screen X:", self.spin_x)
        form.addRow("Screen Y:", self.spin_y)
        form.addRow("Hotkeys:", self.btn_set_pos_key)
        form.addRow("Threshold:", self.spin_threshold)
        form.addRow("Sensitivity:", self.spin_sens)
        form.addRow("Drink Potions:", self.chk_potions)
        form.addRow("Drink Delay (s):", self.spin_potion_delay)
        left_layout.addLayout(form)

        # Buttons
        self.btn_start = QPushButton("Start Fishing")
        self.btn_start.clicked.connect(self._toggle_fishing)
        left_layout.addWidget(self.btn_start)

        self.btn_hotkey_main = QPushButton(f"Change Start Hotkey ({self.hotkey_fishing})")
        self.btn_hotkey_main.clicked.connect(lambda: self._wait_for_hotkey('main'))
        left_layout.addWidget(self.btn_hotkey_main)

        self.btn_save = QPushButton("Save Preset")
        self.btn_save.clicked.connect(self._save_current_preset)
        left_layout.addWidget(self.btn_save)

        # Right side (Presets)
        self.list_presets = QListWidget()
        self.list_presets.itemClicked.connect(self._load_selected_preset)
        right_layout.addWidget(self.list_presets)

        btn_add = QPushButton("New Preset")
        btn_add.clicked.connect(self._add_preset)
        btn_del = QPushButton("Delete Preset")
        btn_del.clicked.connect(self._del_preset)
        right_layout.addWidget(btn_add)
        right_layout.addWidget(btn_del)

    def _create_preview_label(self):
        l = QLabel()
        l.setFixedSize(self.SHIFT * 2, self.SHIFT * 2)
        l.setFrameShape(QFrame.Shape.Box)
        return l

    def _handle_global_hotkey(self, key_char):
        """Слот, принимающий сигнал из потока pynput"""
        if key_char == self.hotkey_fishing:
            self._toggle_fishing()
        elif key_char == self.hotkey_pos:
            pos = pyautogui.position()
            self.spin_x.setValue(pos.x)
            self.spin_y.setValue(pos.y)

    def _toggle_fishing(self):
        if self.fisher_logic:
            self.fisher_logic = None
            self.btn_start.setText("Start Fishing")
            self._set_controls_enabled(True)
            self.lbl_status_state.setText("State: STOPPED")
        else:
            self.fisher_logic = FisherLogic()
            self.potion_timer = time.time()
            self.btn_start.setText("Stop Fishing")
            self._set_controls_enabled(False)

    def _game_loop(self):
        # 1. Capture
        cx, cy = self.spin_x.value(), self.spin_y.value()
        bbox = (cx - self.SHIFT, cy - self.SHIFT, cx + self.SHIFT, cy + self.SHIFT)

        try:
            # Grab only the needed region
            pil_img = ImageGrab.grab(bbox=bbox)
        except OSError:
            return  # Screen lock handling

        # 2. Process
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        threshold = self.spin_threshold.value()
        diff_img = self.tracker.get_diff(gray, threshold)

        # Calculate sensitivity
        count = cv2.countNonZero(diff_img)
        area = (self.SHIFT * 2) ** 2
        sensitivity_mult = self.spin_sens.value()
        sense_val = count * sensitivity_mult / area

        # 3. Update Logic
        if self.fisher_logic:
            current_state = self.fisher_logic.update(sense_val)
            self.lbl_status_state.setText(f"State: {current_state.value}")
            self._handle_potions()
        else:
            m_pos = QCursor.pos()
            self.lbl_status_mouse.setText(f"Mouse: {m_pos.x()}, {m_pos.y()}")

        # 4. Update UI (Visuals)
        self.progress_bar.setValue(min(100, int(sense_val * 100)))

        # Show Raw
        self.lbl_raw.setPixmap(QPixmap.fromImage(ImageQt.ImageQt(pil_img)))

        # Show Diff
        h, w = diff_img.shape
        q_diff = QImage(diff_img.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.lbl_proc.setPixmap(QPixmap.fromImage(q_diff))

    def _handle_potions(self):
        if not self.chk_potions.isChecked():
            self.lbl_status_potion.setText("Potions: OFF")
            return

        delay = self.spin_potion_delay.value()
        time_left = int((self.potion_timer + delay) - time.time())

        if time_left <= 0:
            btn = self.config[self._get_current_preset_name()].get('button_to_drink', 'b')
            MouseController.press_key(btn)
            self.potion_timer = time.time()
            self.lbl_status_potion.setText("Potions: Drank!")
        else:
            self.lbl_status_potion.setText(f"Drink in: {time_left}s")

    # --- Config & Helper Methods ---
    def _set_controls_enabled(self, val):
        self.spin_x.setEnabled(val)
        self.spin_y.setEnabled(val)
        self.list_presets.setEnabled(val)
        # ... disable other inputs ...

    def _load_initial_config(self):
        self.config.read('config.ini')
        if not self.config.sections():
            self.config['DEFAULT'] = {
                'screen_x': '800', 'screen_y': '600',
                'treshold': '6', 'sensivity': '55'
            }
        self._refresh_preset_list()

    def _refresh_preset_list(self):
        self.list_presets.clear()
        for section in self.config.sections():
            self.list_presets.addItem(section)
        self.list_presets.setCurrentRow(0)

    def _get_current_preset_name(self):
        item = self.list_presets.currentItem()
        return item.text() if item else 'DEFAULT'

    def _load_selected_preset(self):
        name = self._get_current_preset_name()
        if name in self.config:
            data = self.config[name]
            self.spin_x.setValue(int(data.get('screen_x', 0)))
            self.spin_y.setValue(int(data.get('screen_y', 0)))
            self.spin_threshold.setValue(int(data.get('treshold', 6)))
            self.spin_sens.setValue(int(data.get('sensivity', 55)))
            self.chk_potions.setChecked(data.get('drink_potions', 'False') == 'True')
            self.spin_potion_delay.setValue(int(data.get('drink_delay', 180)))

    def _save_current_preset(self):
        name = self._get_current_preset_name()
        self.config[name] = {
            'screen_x': str(self.spin_x.value()),
            'screen_y': str(self.spin_y.value()),
            'treshold': str(self.spin_threshold.value()),
            'sensivity': str(self.spin_sens.value()),
            'drink_potions': str(self.chk_potions.isChecked()),
            'drink_delay': str(self.spin_potion_delay.value())
        }
        with open('config.ini', 'w') as f:
            self.config.write(f)

    def _add_preset(self):
        text, ok = QInputDialog.getText(self, 'New Preset', 'Name:')
        if ok and text:
            self.config[text] = {}
            self._save_current_preset()  # Save defaults to it
            self._refresh_preset_list()

    def _del_preset(self):
        name = self._get_current_preset_name()
        if name == 'DEFAULT':
            return
        self.config.remove_section(name)
        with open('config.ini', 'w') as f:
            self.config.write(f)
        self._refresh_preset_list()

    def _wait_for_hotkey(self, target):
        """Dialog to capture a single keypress"""
        d = QDialog(self)
        d.setWindowTitle("Press a key...")
        d.resize(200, 100)

        def key_handler(event):
            key = event.text()
            if key:
                if target == 'main':
                    self.hotkey_fishing = key
                    self.btn_hotkey_main.setText(f"Change Start ({key})")
                elif target == 'pos':
                    self.hotkey_pos = key
                    self.btn_set_pos_key.setText(f"Set Pos ({key})")
                d.accept()

        d.keyPressEvent = key_handler
        d.exec()


def main():
    app = QApplication(sys.argv)
    window = AppUi()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()