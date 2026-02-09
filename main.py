#!/usr/bin/env python3

import sys
import time
import configparser
import enum
import logging
import os
import numpy as np
import cv2
import pyautogui
import sounddevice as sd
import soundfile as sf
from scipy import signal

# PyQt Imports
from PyQt6.QtCore import (Qt, QTimer, pyqtSignal, QObject, QThread)
from PyQt6.QtGui import QPixmap, QImage, QCursor, QAction
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QCheckBox, QFrame, QFormLayout, QHBoxLayout,
                             QVBoxLayout, QSpinBox, QPushButton,
                             QMessageBox, QProgressBar, QListWidget,
                             QInputDialog, QDialog, QTabWidget, QSlider,
                             QComboBox, QGroupBox)

# Pynput & PIL
from PIL import ImageGrab, ImageQt
from pynput.keyboard import Listener, KeyCode

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

__version__ = '1.2 (Pro Configs)'
__author__ = 'Alexei Metlitski'


# ==========================================
#               WORKER THREADS
# ==========================================

class VisualWorker(QThread):
    image_processed = pyqtSignal(QImage, QImage)
    stats_updated = pyqtSignal(float, str)
    potion_drank = pyqtSignal(int)

    def __init__(self, config_data):
        super().__init__()
        self.running = True
        self.cfg = config_data
        self.tracker = MovementTracker()
        self.logic = FisherLogic()
        self.potion_timer = time.time()

    def update_config(self, new_config):
        self.cfg = new_config

    def run(self):
        while self.running:
            # 1. Capture
            cx, cy = int(self.cfg['x']), int(self.cfg['y'])
            shift = 50
            bbox = (cx - shift, cy - shift, cx + shift, cy + shift)

            try:
                pil_img = ImageGrab.grab(bbox=bbox)
            except OSError:
                time.sleep(0.1);
                continue

            # 2. Process
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            diff_img = self.tracker.get_diff(gray, int(self.cfg['threshold']))

            count = cv2.countNonZero(diff_img)
            area = (shift * 2) ** 2
            sense_val = count * int(self.cfg['sensitivity']) / area

            state_desc = self.logic.update(sense_val)

            self._handle_potions()

            # 3. Emit
            q_raw = ImageQt.ImageQt(pil_img)
            h, w = diff_img.shape
            q_proc = QImage(diff_img.data, w, h, w, QImage.Format.Format_Grayscale8)

            self.image_processed.emit(q_raw, q_proc)
            self.stats_updated.emit(sense_val, state_desc)
            time.sleep(0.05)

    def _handle_potions(self):
        if self.cfg['use_potions']:
            delay = int(self.cfg['potion_delay'])
            time_left = int((self.potion_timer + delay) - time.time())
            if time_left <= 0:
                MouseController.press_vk(66)  # 'B'
                self.potion_timer = time.time()
                self.potion_drank.emit(0)
            else:
                self.potion_drank.emit(time_left)
        else:
            self.potion_drank.emit(-1)

    def stop(self):
        self.running = False
        self.wait()


class AudioMonitorWorker(QThread):
    """
    Lightweight thread just to show volume when bot is IDLE.
    Does NOT trigger actions.
    """
    current_volume = pyqtSignal(int)

    def __init__(self, device_idx, gain=1.0):
        super().__init__()
        self.running = True
        self.device_idx = device_idx
        self.gain = gain

    def update_settings(self, device_idx, gain):
        self.device_idx = device_idx
        self.gain = gain

    def run(self):
        def callback(indata, frames, time, status):
            if not self.running: raise sd.CallbackStop()
            vol = np.max(np.abs(indata)) * self.gain
            self.current_volume.emit(int(min(100, vol * 100)))

        try:
            # Just listen at 44100 default
            with sd.InputStream(device=self.device_idx, channels=1, callback=callback, blocksize=2048):
                while self.running:
                    sd.sleep(100)
        except:
            self.current_volume.emit(0)

    def stop(self):
        self.running = False
        self.wait()


class AudioPatternWorker(QThread):
    """
    Heavy worker: Pattern Matching + Logic + Clicking.
    Runs only when bot is STARTed.
    """
    match_score = pyqtSignal(int)
    current_volume = pyqtSignal(int)
    trigger_fired = pyqtSignal(str)
    state_updated = pyqtSignal(str)
    potion_drank = pyqtSignal(int)

    def __init__(self, config_data, device_idx):
        super().__init__()
        self.running = True
        self.cfg = config_data
        self.device_idx = device_idx
        self.potion_timer = time.time()
        self.template = None
        self.stream_rate = 44100

        # Init Template
        try:
            try:
                dev_info = sd.query_devices(self.device_idx, 'input')
                self.stream_rate = int(dev_info['default_samplerate'])
            except:
                self.stream_rate = 44100

            if not os.path.exists('Splash_1.wav'):
                self.state_updated.emit("Error: Splash_1.wav missing!")
                self.running = False
                return

            data, file_rate = sf.read('Splash_1.wav')
            if len(data.shape) > 1: data = data.mean(axis=1)

            # Auto-Resample
            if file_rate != self.stream_rate:
                self.state_updated.emit(f"Resampling {file_rate}->{self.stream_rate}...")
                number_of_samples = int(round(len(data) * float(self.stream_rate) / file_rate))
                data = signal.resample(data, number_of_samples)

            self.template = data / np.max(np.abs(data))
            self.state_updated.emit(f"Ready ({self.stream_rate}Hz)")

        except Exception as e:
            self.state_updated.emit(f"Init Error: {e}")
            self.running = False

    def update_config(self, new_config):
        self.cfg = new_config

    def run(self):
        if not self.running or self.template is None: return
        block_size = len(self.template)

        def callback(indata, frames, time_info, status):
            if not self.running: raise sd.CallbackStop()

            try:
                live_audio = indata.flatten()

                # Apply Gain
                gain = float(self.cfg['audio_gain'])
                live_audio = live_audio * gain

                # Calc Volume
                vol_peak = np.max(np.abs(live_audio))
                self.current_volume.emit(int(min(100, vol_peak * 100)))

                if vol_peak < 0.001:
                    self.match_score.emit(0)
                    return

                # Pattern Match
                live_audio = live_audio / vol_peak  # Normalize
                correlation = signal.correlate(live_audio, self.template, mode='same', method='fft')
                peak = np.max(np.abs(correlation))
                score = int(min(100, (peak / len(self.template)) * 100 * 2.5))
                self.match_score.emit(score)

                # Trigger Logic
                if score > int(self.cfg['audio_threshold']):
                    self.trigger_fired.emit("MATCH!")
                    self.state_updated.emit(f"SPLASH! ({score}%)")

                    MouseController.click()
                    sd.sleep(2500)
                    MouseController.click()
                    sd.sleep(1500)
                    self.state_updated.emit("Listening...")

            except Exception as e:
                print(f"Proc: {e}")

        try:
            with sd.InputStream(device=self.device_idx, channels=1, callback=callback,
                                blocksize=block_size, samplerate=self.stream_rate):
                while self.running:
                    self._handle_potions()
                    sd.sleep(100)
        except Exception as e:
            self.state_updated.emit(f"Stream Error: {e}")

    def _handle_potions(self):
        if self.cfg['use_potions']:
            delay = int(self.cfg['potion_delay'])
            time_left = int((self.potion_timer + delay) - time.time())
            if time_left <= 0:
                MouseController.press_vk(66);
                self.potion_timer = time.time();
                self.potion_drank.emit(0)
            else:
                self.potion_drank.emit(time_left)
        else:
            self.potion_drank.emit(-1)

    def stop(self):
        self.running = False; self.wait()


# ==========================================
#               LOGIC HELPERS
# ==========================================

class KeyMonitor(QObject):
    keyPressed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        try:
            vk = key.vk if hasattr(key, 'vk') else key.value.vk
            if vk: self.keyPressed.emit(vk)
        except:
            pass


class MouseController:
    @staticmethod
    def click():
        pyautogui.mouseDown()
        time.sleep(0.05)
        pyautogui.mouseUp()

    @staticmethod
    def press_vk(vk_code):
        try:
            char = KeyCode.from_vk(vk_code).char
            if char: pyautogui.press(char)
        except:
            pass


class FisherLogic:
    def __init__(self):
        self.state = "INIT"
        self.last_action = time.time()

    def update(self, sense_level: float) -> str:
        now = time.time()
        if self.state == "INIT":
            if sense_level > 1: self._switch("CAST"); MouseController.click()
        elif self.state == "CAST":
            if (now - self.last_action) > 1.5 and sense_level < 1: self._switch("WAIT")
        elif self.state == "WAIT":
            if (now - self.last_action) > 1.0 and sense_level > 1: self._switch("REEL"); MouseController.click()
        elif self.state == "REEL":
            if (now - self.last_action) > 2.0 and sense_level < 1: self._switch("CAST"); MouseController.click()
        return self.state

    def _switch(self, new_state):
        self.state = new_state;
        self.last_action = time.time()


class MovementTracker:
    def __init__(self, size=3):
        self.buffer = [];
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


# ==========================================
#               MAIN GUI
# ==========================================

class AppUi(QMainWindow):
    STYLE_GREEN = "QPushButton { background-color: #4caf50; color: black; font-weight: bold; border-radius: 4px; }"
    STYLE_RED = "QPushButton { background-color: #d32f2f; color: white; font-weight: bold; border-radius: 4px; }"
    CONFIG_FILE = 'fishing_profiles.ini'

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'AutoFisher v{__version__}')
        self.resize(450, 650)

        self.config = configparser.ConfigParser()
        self.active_worker = None
        self.monitor_worker = None
        self.hotkeys_active = True

        # Hotkeys VK Defaults
        self.vk_fishing = 70;
        self.txt_fishing = 'F'
        self.vk_pos = 86;
        self.txt_pos = 'V'

        self._init_ui()
        self._load_profiles_from_file()  # Populate dropdown
        self._start_monitor()  # Start volume meter immediately

        self.key_monitor = KeyMonitor()
        self.key_monitor.keyPressed.connect(self._handle_hotkey_vk)

    def _init_ui(self):
        central = QWidget();
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- 1. CONFIGURATION GROUP ---
        config_group = QGroupBox("Profile Management")
        cg_layout = QVBoxLayout(config_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Profile:"))
        self.combo_profiles = QComboBox()
        self.combo_profiles.currentIndexChanged.connect(self._on_profile_changed)
        row1.addWidget(self.combo_profiles, 1)

        btn_new = QPushButton("New");
        btn_new.clicked.connect(self._new_profile)
        btn_del = QPushButton("Del");
        btn_del.clicked.connect(self._del_profile)
        btn_save = QPushButton("Save");
        btn_save.clicked.connect(self._save_current_profile)

        row1.addWidget(btn_new);
        row1.addWidget(btn_del);
        row1.addWidget(btn_save)
        cg_layout.addLayout(row1)
        main_layout.addWidget(config_group)

        # --- 2. GLOBAL HOTKEY BTN ---
        self.btn_hotkeys = QPushButton("HOTKEYS: ON")
        self.btn_hotkeys.clicked.connect(self._toggle_hotkeys_state)
        self._refresh_hotkey_btn()
        main_layout.addWidget(self.btn_hotkeys)

        # --- 3. TABS ---
        self.tabs = QTabWidget()
        self.tab_visual = QWidget()
        self.tab_audio = QWidget()
        self.tabs.addTab(self.tab_visual, "Visual Mode")
        self.tabs.addTab(self.tab_audio, "Audio Pattern Mode")

        # Connect tab change to restart monitor (optional, but good for switching context)
        # self.tabs.currentChanged.connect(self._restart_monitor)

        self._init_visual_tab()
        self._init_audio_tab()
        main_layout.addWidget(self.tabs)

        # --- 4. FOOTER (Shared) ---
        footer = QVBoxLayout()
        pot_layout = QHBoxLayout()
        self.chk_pot = QCheckBox("Drink Potions")
        self.chk_pot.stateChanged.connect(self._update_runtime_config)
        self.spin_delay = QSpinBox();
        self.spin_delay.setRange(0, 3600);
        self.spin_delay.setSuffix("s")
        self.spin_delay.valueChanged.connect(self._update_runtime_config)
        self.lbl_potion = QLabel("Potions: OFF")
        pot_layout.addWidget(self.chk_pot);
        pot_layout.addWidget(self.spin_delay);
        pot_layout.addWidget(self.lbl_potion)
        footer.addLayout(pot_layout)

        self.btn_start = QPushButton(f"Start Fishing ({self.txt_fishing})")
        self.btn_start.setFixedHeight(40);
        self.btn_start.setStyleSheet(self.STYLE_GREEN)
        self.btn_start.clicked.connect(self._toggle_active_worker)
        footer.addWidget(self.btn_start)

        hk_layout = QHBoxLayout()
        self.btn_main_key = QPushButton("Change Start Key");
        self.btn_main_key.clicked.connect(lambda: self._wait_key('main'))
        hk_layout.addWidget(self.btn_main_key)
        footer.addLayout(hk_layout)
        main_layout.addLayout(footer)

    def _init_visual_tab(self):
        layout = QVBoxLayout(self.tab_visual)
        prev_layout = QHBoxLayout()
        self.lbl_raw = self._create_preview("Raw");
        self.lbl_proc = self._create_preview("Mask")
        prev_layout.addWidget(self.lbl_raw);
        prev_layout.addWidget(self.lbl_proc)
        layout.addLayout(prev_layout)

        self.vis_prog_bar = QProgressBar()
        self.lbl_vis_state = QLabel("State: IDLE")
        layout.addWidget(self.vis_prog_bar);
        layout.addWidget(self.lbl_vis_state)

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

        for w in [self.spin_x, self.spin_y, self.spin_th, self.spin_sens]:
            w.valueChanged.connect(self._update_runtime_config)

        form.addRow("X / Y:", self._h_box(self.spin_x, self.spin_y))
        form.addRow("Pos Hotkey:", self.btn_pos_key)
        form.addRow("Threshold:", self.spin_th);
        form.addRow("Sensitivity:", self.spin_sens)
        layout.addLayout(form)

    def _init_audio_tab(self):
        layout = QVBoxLayout(self.tab_audio)

        layout.addWidget(QLabel("Select Loopback Device:"))
        self.combo_devices = QComboBox()
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    self.combo_devices.addItem(f"{idx}: {dev['name']}", idx)
        except:
            pass
        self.combo_devices.currentIndexChanged.connect(self._restart_monitor)
        layout.addWidget(self.combo_devices)

        # Gain Slider
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Digital Gain:"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(10, 100)  # 1.0 to 10.0
        self.slider_gain.setValue(10)
        self.lbl_gain_val = QLabel("1.0x")
        self.slider_gain.valueChanged.connect(
            lambda v: (self.lbl_gain_val.setText(f"{v / 10:.1f}x"), self._update_runtime_config(),
                       self._update_monitor_gain()))
        gain_layout.addWidget(self.slider_gain)
        gain_layout.addWidget(self.lbl_gain_val)
        layout.addLayout(gain_layout)

        # Visualizers
        layout.addWidget(QLabel("1. Raw Volume:"))
        self.volume_bar = QProgressBar();
        self.volume_bar.setRange(0, 100);
        self.volume_bar.setFormat("%v %")
        self.volume_bar.setStyleSheet("QProgressBar::chunk { background-color: #2196F3; }")
        layout.addWidget(self.volume_bar)

        layout.addWidget(QLabel("2. Match Score:"))
        self.audio_bar = QProgressBar();
        self.audio_bar.setRange(0, 100);
        self.audio_bar.setFormat("%v %")
        self.audio_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        layout.addWidget(self.audio_bar)

        self.lbl_audio_state = QLabel("Status: Ready")
        layout.addWidget(self.lbl_audio_state)

        # Threshold
        form = QFormLayout()
        self.slider_audio_th = QSlider(Qt.Orientation.Horizontal);
        self.slider_audio_th.setRange(0, 100);
        self.slider_audio_th.setValue(60)
        self.lbl_th_val = QLabel("60%")
        self.slider_audio_th.valueChanged.connect(
            lambda v: (self.lbl_th_val.setText(str(v) + "%"), self._update_runtime_config()))
        form.addRow("Match Threshold:", self._h_box(self.slider_audio_th, self.lbl_th_val))
        layout.addLayout(form)

    def _h_box(self, w1, w2):
        w = QWidget();
        l = QHBoxLayout(w);
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(w1);
        l.addWidget(w2);
        return w

    def _create_preview(self, t):
        l = QLabel();
        l.setFixedSize(100, 100);
        l.setFrameShape(QFrame.Shape.Box);
        l.setToolTip(t);
        return l

    # ==========================================
    #           PROFILE MANAGEMENT
    # ==========================================

    def _load_profiles_from_file(self):
        self.config.read(self.CONFIG_FILE)
        if not self.config.sections():
            self.config['Default'] = {
                'x': '0', 'y': '0', 'vis_th': '10', 'vis_sens': '50',
                'potions': 'False', 'pot_delay': '180',
                'audio_th': '60', 'audio_gain': '1.0'
            }
            with open(self.CONFIG_FILE, 'w') as f: self.config.write(f)

        self.combo_profiles.blockSignals(True)
        self.combo_profiles.clear()
        for s in self.config.sections():
            self.combo_profiles.addItem(s)
        self.combo_profiles.blockSignals(False)
        self._on_profile_changed()

    def _on_profile_changed(self):
        name = self.combo_profiles.currentText()
        if name in self.config:
            data = self.config[name]
            # Visual
            self.spin_x.setValue(int(data.get('x', 0)))
            self.spin_y.setValue(int(data.get('y', 0)))
            self.spin_th.setValue(int(data.get('vis_th', 10)))
            self.spin_sens.setValue(int(data.get('vis_sens', 50)))
            # Audio
            self.slider_audio_th.setValue(int(data.get('audio_th', 60)))
            gain_val = float(data.get('audio_gain', 1.0))
            self.slider_gain.setValue(int(gain_val * 10))
            # Shared
            self.chk_pot.setChecked(data.get('potions', 'False') == 'True')
            self.spin_delay.setValue(int(data.get('pot_delay', 180)))

    def _save_current_profile(self):
        name = self.combo_profiles.currentText()
        if not name: return

        self.config[name] = {
            'x': str(self.spin_x.value()),
            'y': str(self.spin_y.value()),
            'vis_th': str(self.spin_th.value()),
            'vis_sens': str(self.spin_sens.value()),
            'audio_th': str(self.slider_audio_th.value()),
            'audio_gain': str(self.slider_gain.value() / 10.0),
            'potions': str(self.chk_pot.isChecked()),
            'pot_delay': str(self.spin_delay.value())
        }

        with open(self.CONFIG_FILE, 'w') as f: self.config.write(f)
        QMessageBox.information(self, "Saved", f"Profile '{name}' saved successfully!")

    def _new_profile(self):
        t, ok = QInputDialog.getText(self, 'New Profile', 'Name:')
        if ok and t:
            self.config[t] = {}  # Empty dict to reserve spot
            self.combo_profiles.addItem(t)
            self.combo_profiles.setCurrentIndex(self.combo_profiles.count() - 1)
            self._save_current_profile()  # Save current UI settings to new profile

    def _del_profile(self):
        name = self.combo_profiles.currentText()
        if self.combo_profiles.count() <= 1: return

        confirm = QMessageBox.question(self, "Delete", f"Delete '{name}'?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            self.config.remove_section(name)
            with open(self.CONFIG_FILE, 'w') as f: self.config.write(f)
            self._load_profiles_from_file()

    # ==========================================
    #           WORKER LOGIC
    # ==========================================

    def _get_current_ui_config(self):
        return {
            'x': self.spin_x.value(), 'y': self.spin_y.value(),
            'threshold': self.spin_th.value(), 'sensitivity': self.spin_sens.value(),
            'audio_threshold': self.slider_audio_th.value(),
            'audio_gain': self.slider_gain.value() / 10.0,
            'use_potions': self.chk_pot.isChecked(),
            'potion_delay': self.spin_delay.value()
        }

    def _update_runtime_config(self):
        if self.active_worker: self.active_worker.update_config(self._get_current_ui_config())

    def _toggle_active_worker(self):
        if self.active_worker:
            # STOPPING
            self.active_worker.stop()
            self.active_worker = None
            self.btn_start.setText(f"Start Fishing ({self.txt_fishing})")
            self.btn_start.setStyleSheet(self.STYLE_GREEN)
            self.tabs.setEnabled(True)
            self.lbl_vis_state.setText("State: STOPPED")
            self.lbl_audio_state.setText("Status: Stopped")

            # Resume Monitor
            self._start_monitor()
        else:
            # STARTING
            # Stop Monitor first to free up audio device
            self._stop_monitor()

            self.tabs.setEnabled(False)
            cfg = self._get_current_ui_config()

            if self.tabs.currentIndex() == 0:
                # Visual
                self.active_worker = VisualWorker(cfg)
                self.active_worker.image_processed.connect(lambda r, p: (self.lbl_raw.setPixmap(QPixmap.fromImage(r)),
                                                                         self.lbl_proc.setPixmap(QPixmap.fromImage(p))))
                self.active_worker.stats_updated.connect(lambda s, t: (self.lbl_vis_state.setText(f"State: {t}"),
                                                                       self.vis_prog_bar.setValue(
                                                                           min(100, int(s * 100)))))
            else:
                # Audio
                dev_idx = self.combo_devices.currentData()
                self.active_worker = AudioPatternWorker(cfg, dev_idx)
                self.active_worker.match_score.connect(self.audio_bar.setValue)
                self.active_worker.current_volume.connect(self.volume_bar.setValue)
                self.active_worker.state_updated.connect(self.lbl_audio_state.setText)

            self.active_worker.potion_drank.connect(self._on_potion_update)
            self.active_worker.start()
            self.btn_start.setText(f"Stop Fishing ({self.txt_fishing})")
            self.btn_start.setStyleSheet(self.STYLE_RED)

    def _on_potion_update(self, t):
        self.lbl_potion.setText(f"Drink in: {t}s" if t > 0 else ("DRANK!" if t == 0 else "OFF"))

    # --- MONITOR LOGIC ---
    def _start_monitor(self):
        if self.monitor_worker: return
        dev_idx = self.combo_devices.currentData()
        gain = self.slider_gain.value() / 10.0
        self.monitor_worker = AudioMonitorWorker(dev_idx, gain)
        self.monitor_worker.current_volume.connect(self.volume_bar.setValue)
        self.monitor_worker.start()

    def _stop_monitor(self):
        if self.monitor_worker:
            self.monitor_worker.stop()
            self.monitor_worker = None
            self.volume_bar.setValue(0)

    def _restart_monitor(self):
        # Restart monitor if device changed, but ONLY if active worker is not running
        if not self.active_worker:
            self._stop_monitor()
            self._start_monitor()

    def _update_monitor_gain(self):
        if self.monitor_worker:
            self.monitor_worker.update_settings(self.combo_devices.currentData(), self.slider_gain.value() / 10.0)

    # --- HOTKEYS ---
    def _toggle_hotkeys_state(self):
        self.hotkeys_active = not self.hotkeys_active
        self._refresh_hotkey_btn()

    def _refresh_hotkey_btn(self):
        if self.hotkeys_active:
            self.btn_hotkeys.setText("HOTKEYS: ON")
            self.btn_hotkeys.setStyleSheet(self.STYLE_RED)
        else:
            self.btn_hotkeys.setText("HOTKEYS: MUTED")
            self.btn_hotkeys.setStyleSheet(self.STYLE_GREEN)

    def _handle_hotkey_vk(self, vk):
        if not self.hotkeys_active: return
        if vk == self.vk_fishing:
            self._toggle_active_worker()
        elif vk == self.vk_pos:
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

        def on_press(key):
            vk = key.vk if hasattr(key, 'vk') else key.value.vk
            char = key.char.upper() if hasattr(key, 'char') and key.char else f"VK_{vk}"
            if vk:
                if target == 'main':
                    self.vk_fishing = vk; self.txt_fishing = char; self.btn_start.setText(f"Start Fishing ({char})")
                elif target == 'pos':
                    self.vk_pos = vk; self.txt_pos = char; self.btn_pos_key.setText(f"Set Pos Key ({char})")
                d.accept();
                return False

        listener = Listener(on_press=on_press);
        listener.start();
        d.exec();
        listener.stop()


def main():
    app = QApplication(sys.argv)
    window = AppUi()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()