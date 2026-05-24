#!/usr/bin/env python3

import sys
import time
import configparser
import logging
import os
import glob
import collections
import numpy as np
import cv2
import pyautogui
import sounddevice as sd
import soundfile as sf
from scipy import signal

# PyQt Imports
from PyQt6.QtCore import (Qt, pyqtSignal, QObject, QThread, QTimer)
from PyQt6.QtGui import QPixmap, QImage, QCloseEvent
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QCheckBox, QFrame, QFormLayout, QHBoxLayout,
                             QVBoxLayout, QSpinBox, QPushButton,
                             QMessageBox, QProgressBar, QTabWidget, QSlider,
                             QComboBox, QGroupBox, QDialog, QInputDialog)

# Pynput & PIL
from PIL import ImageGrab, ImageQt
from pynput import mouse
from pynput.keyboard import Listener, KeyCode, Controller as KeyboardController

_keyboard_ctrl = KeyboardController()

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

__version__ = '2'
__author__ = 'Yehor Bondarchuk'


# ==========================================
#          RESOURCE & PATH HELPERS
# ==========================================

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def get_app_data_dir():
    app_data = os.getenv('APPDATA')
    path = os.path.join(app_data, 'AutoFisher', 'Patterns')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def tick_potion(cfg, potion_timer):
    """Returns (new_timer, signal_value). Signal value: -1 disabled, 0 just drank, >0 seconds left."""
    if not cfg.get('use_potions'):
        return potion_timer, -1
    delay = int(cfg['potion_delay'])
    time_left = int((potion_timer + delay) - time.time())
    if time_left <= 0:
        MouseController.press_vk(cfg.get('potion_vk', 66))
        return time.time(), 0
    return potion_timer, time_left


def cleanup_temp_patterns():
    folder = get_app_data_dir()
    for f in glob.glob(os.path.join(folder, "Splash_*.wav")):
        try:
            os.remove(f)
        except:
            pass


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
        self.logic = FisherLogic(timeout=float(config_data.get('visual_timeout', 10)))
        self.potion_timer = time.time()

    def update_config(self, new_config):
        self.cfg = new_config
        try:
            self.logic.timeout = float(new_config.get('visual_timeout', 10))
        except (TypeError, ValueError):
            pass

    def run(self):
        while self.running:
            cx, cy = int(self.cfg['x']), int(self.cfg['y'])
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

            diff_img = self.tracker.get_diff(gray, int(self.cfg['threshold']))

            count = cv2.countNonZero(diff_img)
            area = (shift * 2) ** 2
            sense_val = count * int(self.cfg['sensitivity']) / area

            state_desc = self.logic.update(sense_val)
            self._handle_potions()

            q_raw = ImageQt.ImageQt(pil_img)
            h, w = diff_img.shape
            q_proc = QImage(diff_img.data, w, h, w, QImage.Format.Format_Grayscale8)

            self.image_processed.emit(q_raw, q_proc)
            self.stats_updated.emit(sense_val, state_desc)
            time.sleep(0.05)

    def _handle_potions(self):
        self.potion_timer, signal_val = tick_potion(self.cfg, self.potion_timer)
        self.potion_drank.emit(signal_val)

    def stop(self):
        self.running = False
        self.wait()


class AudioMonitorWorker(QThread):
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
            with sd.InputStream(device=self.device_idx, channels=1, callback=callback, blocksize=4096):
                while self.running:
                    sd.sleep(100)
        except:
            self.current_volume.emit(0)

    def stop(self):
        self.running = False
        self.wait()


class AudioTrainerWorker(QThread):
    pattern_saved = pyqtSignal(int)
    training_log = pyqtSignal(str)
    volume_level = pyqtSignal(int)

    def __init__(self, device_idx, gain=1.0):
        super().__init__()
        self.running = True
        self.device_idx = device_idx
        self.gain = gain
        self.buffer = collections.deque(maxlen=40)
        self.samplerate = 44100
        self.mouse_listener = None
        self.click_count = 0

    def run(self):
        try:
            dev_info = sd.query_devices(self.device_idx, 'input')
            self.samplerate = int(dev_info['default_samplerate'])
        except:
            pass

        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.mouse_listener.start()

        def callback(indata, frames, time, status):
            if not self.running: raise sd.CallbackStop()
            data = indata.flatten() * self.gain
            self.buffer.append(data.copy())
            vol = np.max(np.abs(data))
            self.volume_level.emit(int(min(100, vol * 100)))

        try:
            with sd.InputStream(device=self.device_idx, channels=1, callback=callback,
                                samplerate=self.samplerate):
                while self.running:
                    sd.sleep(50)
        except Exception as e:
            self.training_log.emit(f"Error: {e}")

    def _on_click(self, x, y, button, pressed):
        if not self.running: return False
        if pressed and button == mouse.Button.left:
            self.click_count += 1
            if self.click_count % 2 != 0:
                self.training_log.emit("Casted... WAIT FOR SPLASH!")
                self.buffer.clear()
            else:
                self._save_next_pattern()
                self.training_log.emit("Caught! Cast again...")

    def _save_next_pattern(self):
        try:
            app_data_path = get_app_data_dir()
            existing = glob.glob(os.path.join(app_data_path, "Splash_*.wav"))
            # Index 1 is the bundled pattern; user-recorded patterns start at 2.
            # Use max(existing)+1 so non-contiguous files (after deletions) don't collide.
            used = []
            for p in existing:
                try:
                    used.append(int(os.path.splitext(os.path.basename(p))[0].split('_')[1]))
                except (ValueError, IndexError):
                    pass
            next_idx = max(used + [1]) + 1

            filename = os.path.join(app_data_path, f"Splash_{next_idx}.wav")

            full_audio = np.concatenate(list(self.buffer))
            duration_samples = int(1.5 * self.samplerate)
            if len(full_audio) > duration_samples:
                full_audio = full_audio[-duration_samples:]

            trimmed = self._smart_trim(full_audio)

            sf.write(filename, trimmed, self.samplerate)
            self.pattern_saved.emit(next_idx)

        except Exception as e:
            self.training_log.emit(f"Save Error: {e}")

    def _smart_trim(self, data):
        peak_idx = np.argmax(np.abs(data))
        pre = int(0.15 * self.samplerate)
        post = int(0.35 * self.samplerate)
        start = max(0, peak_idx - pre)
        end = min(len(data), peak_idx + post)
        return data[start:end]

    def stop(self):
        self.running = False
        if self.mouse_listener: self.mouse_listener.stop()
        self.wait()


class AudioPatternWorker(QThread):
    match_score = pyqtSignal(int)
    current_volume = pyqtSignal(int)
    trigger_fired = pyqtSignal(str)
    state_updated = pyqtSignal(str)
    potion_drank = pyqtSignal(int)

    def __init__(self, config_data, device_idx):
        super().__init__()
        self.running = True
        self.is_stopping = False
        self.cfg = config_data
        self.device_idx = device_idx
        self.potion_timer = time.time()

        self.templates = []
        self.device_rate = 44100
        self.process_rate = 8000

        # Trigger is requested from the audio callback and executed by run().
        # Cooldown window suppresses re-fires while the click sequence runs.
        self._trigger_pending = False
        self._cooldown_until = 0.0
        self._trigger_action_sec = 4.0  # click + 2.5s + click + 1.5s

        self.sos_filter = None  # built in _prepare_templates

    def _prepare_templates(self):
        """Heavy setup: probe device, load WAVs, build filter. Runs on the worker thread."""
        try:
            try:
                dev_info = sd.query_devices(self.device_idx, 'input')
                self.device_rate = int(dev_info['default_samplerate'])
            except:
                self.device_rate = 44100

            bundled_path = resource_path('Splash_1.wav')
            temp_path = get_app_data_dir()
            temp_files = glob.glob(os.path.join(temp_path, "Splash_*.wav"))

            all_files = []
            if os.path.exists(bundled_path):
                all_files.append(bundled_path)
            all_files.extend(temp_files)

            if not all_files:
                self.state_updated.emit("Error: No Splash_1.wav found!")
                self.running = False
                return

            self.sos_filter = signal.butter(10, 1000, 'hp', fs=self.process_rate, output='sos')

            count = 0
            for p_file in all_files:
                try:
                    data, file_rate = sf.read(p_file)
                    if len(data.shape) > 1: data = data.mean(axis=1)

                    data = self._smart_trim(data, file_rate)
                    target_len = int(len(data) * self.process_rate / file_rate)
                    data = signal.resample(data, target_len)

                    filtered_data = signal.sosfilt(self.sos_filter, data)
                    env = self._get_rms_envelope(filtered_data)

                    noise_floor = np.min(env)
                    env = env - noise_floor
                    max_val = np.max(env)
                    if max_val > 0:
                        env = env / max_val

                    self.templates.append(env)
                    count += 1
                except:
                    pass

            self.state_updated.emit(f"Ready ({count} patterns)")

        except Exception as e:
            self.state_updated.emit(f"Init Error: {e}")
            self.running = False

    def update_config(self, new_config):
        self.cfg = new_config

    def _smart_trim(self, data, rate, duration_sec=0.4):
        if len(data) <= rate * duration_sec: return data
        peak_idx = np.argmax(np.abs(data))
        half_window = int((rate * duration_sec) / 2)
        start = max(0, peak_idx - half_window)
        end = min(len(data), peak_idx + half_window)
        return data[start:end]

    def _get_rms_envelope(self, data, window_size=100):
        squared = np.power(data, 2)
        window = np.ones(window_size) / window_size
        mean_squared = np.convolve(squared, window, mode='same')
        return np.sqrt(mean_squared)

    def run(self):
        self._prepare_templates()
        if not self.running or not self.templates: return

        max_len = max([len(t) for t in self.templates])
        template_duration = max_len / self.process_rate
        block_size = int(template_duration * self.device_rate)
        downsample_factor = int(self.device_rate / self.process_rate)

        def callback(indata, frames, time_info, status):
            if not self.running or self.is_stopping:
                raise sd.CallbackStop()
            try:
                # Skip matching during the click-sequence cooldown window.
                if time.time() < self._cooldown_until:
                    return

                live_audio = indata.flatten()
                gain = float(self.cfg['audio_gain'])
                live_audio = live_audio * gain

                vol_peak = np.max(np.abs(live_audio))
                if not self.is_stopping:
                    self.current_volume.emit(int(min(100, vol_peak * 100)))

                if vol_peak < 0.02:
                    if not self.is_stopping: self.match_score.emit(0)
                    return

                live_downsampled = live_audio[::downsample_factor]
                live_filtered = signal.sosfilt(self.sos_filter, live_downsampled)
                live_envelope = self._get_rms_envelope(live_filtered)

                local_noise = np.min(live_envelope)
                live_envelope = live_envelope - local_noise

                live_max = np.max(live_envelope)
                if live_max > 0.001:
                    live_envelope = live_envelope / live_max

                best_score = 0
                for temp in self.templates:
                    correlation = signal.correlate(live_envelope, temp, mode='valid', method='fft')
                    if correlation.size == 0:
                        correlation = signal.correlate(live_envelope, temp, mode='same', method='fft')

                    peak = np.max(correlation)
                    score = int(min(100, (peak / len(temp)) * 100 * 1.2))
                    if score > best_score:
                        best_score = score

                if not self.is_stopping:
                    self.match_score.emit(best_score)

                    if best_score > int(self.cfg['audio_threshold']):
                        # Hand off to run() — never block inside a PortAudio callback.
                        self._cooldown_until = time.time() + self._trigger_action_sec + 0.5
                        self._trigger_pending = True
                        self.trigger_fired.emit("MATCH!")
                        self.state_updated.emit(f"SPLASH! ({best_score}%)")
            except Exception:
                pass

        try:
            with sd.InputStream(device=self.device_idx, channels=1, callback=callback,
                                blocksize=block_size, samplerate=self.device_rate):
                while self.running:
                    if self.is_stopping: break
                    if self._trigger_pending:
                        self._trigger_pending = False
                        self._do_catch_sequence()
                    self._handle_potions()
                    sd.sleep(100)
        except Exception as e:
            if not self.is_stopping:
                self.state_updated.emit(f"Stream Error: {e}")

    def _do_catch_sequence(self):
        # Reel in, wait for the bobber to re-cast, click again. Must run on the
        # worker thread, not the audio callback.
        MouseController.click()
        self._interruptible_sleep(2.5)
        if self.is_stopping or not self.running: return
        MouseController.click()
        self._interruptible_sleep(1.5)
        if not self.is_stopping:
            self.state_updated.emit("Listening...")

    def _interruptible_sleep(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            if self.is_stopping or not self.running: return
            time.sleep(0.05)

    def _handle_potions(self):
        self.potion_timer, signal_val = tick_potion(self.cfg, self.potion_timer)
        self.potion_drank.emit(signal_val)

    def stop(self):
        self.is_stopping = True
        self.running = False
        self.wait()


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

    def stop(self):
        self.listener.stop()


class MouseController:
    @staticmethod
    def click():
        pyautogui.mouseDown()
        time.sleep(0.05)
        pyautogui.mouseUp()

    @staticmethod
    def press_vk(vk_code):
        # Use pynput Controller so any VK works, not just keys with a .char
        try:
            key = KeyCode.from_vk(int(vk_code))
            _keyboard_ctrl.press(key)
            time.sleep(0.03)
            _keyboard_ctrl.release(key)
        except Exception as e:
            logging.warning(f"press_vk({vk_code}) failed: {e}")


class FisherLogic:
    def __init__(self, timeout=10.0):
        self.state = "INIT"
        self.last_action = time.time()
        self.timeout = timeout

    def update(self, sense_level: float) -> str:
        now = time.time()

        # --- TIMEOUT LOGIC (NEW) ---
        # If waiting too long (10s), force a reset (Reel -> Cast)
        if self.state == "WAIT" and (now - self.last_action) > self.timeout:
            self._switch("REEL")  # Reel in the empty hook
            MouseController.click()
            return "TIMEOUT! Resetting..."
        # ---------------------------

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
    STYLE_ORANGE = "QPushButton { background-color: #ff9800; color: black; font-weight: bold; border-radius: 4px; }"
    CONFIG_FILE = 'fishing_profiles.ini'

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'AutoFisher v{__version__}')
        self.resize(450, 680)

        self.config = configparser.ConfigParser()
        self.active_worker = None
        self.monitor_worker = None
        self.train_worker = None
        self.hotkeys_active = True
        self.loading_profile = False  # LOCK FLAG
        self.rebinding_key = False  # suppresses hotkeys while _wait_key dialog is open

        self.vk_fishing = 70;
        self.txt_fishing = 'F'
        self.vk_pos = 86;
        self.txt_pos = 'V'
        self.vk_train = 84;
        self.txt_train = 'T'
        self.vk_potion = 66;
        self.txt_potion = 'B'

        self._init_ui()
        self._load_profiles_from_file()
        self._start_monitor()

        self.key_monitor = KeyMonitor()
        self.key_monitor.keyPressed.connect(self._handle_hotkey_vk)
        self._update_pattern_count()

    def closeEvent(self, event: QCloseEvent):
        # Stop input first so no new events arrive during teardown.
        self.hotkeys_active = False
        if self.key_monitor:
            try: self.key_monitor.stop()
            except Exception: pass
        # Each worker's stop() already calls wait(); shut down all three.
        for w in (self.active_worker, self.train_worker, self.monitor_worker):
            if w is None: continue
            try: w.stop()
            except Exception as e: logging.warning(f"Worker stop failed: {e}")
        event.accept()

    def _init_ui(self):
        central = QWidget();
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 1. PROFILE
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

        # 2. HOTKEYS
        self.btn_hotkeys = QPushButton("HOTKEYS: ON")
        self.btn_hotkeys.clicked.connect(self._toggle_hotkeys_state)
        self._refresh_hotkey_btn()
        main_layout.addWidget(self.btn_hotkeys)

        # 3. TABS
        self.tabs = QTabWidget()
        self.tab_visual = QWidget()
        self.tab_audio = QWidget()
        self.tabs.addTab(self.tab_visual, "Visual Mode")
        self.tabs.addTab(self.tab_audio, "Audio Pattern Mode")
        self._init_visual_tab()
        self._init_audio_tab()
        main_layout.addWidget(self.tabs)

        # 4. FOOTER
        footer = QVBoxLayout()
        pot_layout = QHBoxLayout()
        self.chk_pot = QCheckBox("Drink Potions")
        self.chk_pot.stateChanged.connect(self._update_runtime_config)
        self.spin_delay = QSpinBox();
        self.spin_delay.setRange(0, 3600);
        self.spin_delay.setSuffix("s")
        self.spin_delay.valueChanged.connect(self._update_runtime_config)
        self.lbl_potion = QLabel("Potions: OFF")
        self.btn_potion_key = QPushButton(f"Key: {self.txt_potion}")
        self.btn_potion_key.setToolTip("Key sent to drink potion")
        self.btn_potion_key.clicked.connect(lambda: self._wait_key('potion'))
        pot_layout.addWidget(self.chk_pot);
        pot_layout.addWidget(self.spin_delay);
        pot_layout.addWidget(self.btn_potion_key);
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
        self.vis_prog_bar = QProgressBar();
        self.lbl_vis_state = QLabel("State: IDLE")
        layout.addWidget(self.vis_prog_bar);
        layout.addWidget(self.lbl_vis_state)
        form = QFormLayout()
        self.spin_x = QSpinBox();
        self.spin_x.setRange(0, 5000);
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
        self.spin_timeout = QSpinBox();
        self.spin_timeout.setRange(2, 120);
        self.spin_timeout.setValue(10);
        self.spin_timeout.setSuffix("s")
        for w in [self.spin_x, self.spin_y, self.spin_th, self.spin_sens, self.spin_timeout]: w.valueChanged.connect(
            self._update_runtime_config)
        form.addRow("X / Y:", self._h_box(self.spin_x, self.spin_y));
        form.addRow("Pos Hotkey:", self.btn_pos_key)
        form.addRow("Threshold:", self.spin_th);
        form.addRow("Sensitivity:", self.spin_sens)
        form.addRow("Re-cast Timeout:", self.spin_timeout)
        layout.addLayout(form)

    def _init_audio_tab(self):
        layout = QVBoxLayout(self.tab_audio)
        layout.addWidget(QLabel("Select Loopback Device:"))
        dev_row = QHBoxLayout()
        self.combo_devices = QComboBox()
        self.combo_devices.currentIndexChanged.connect(self._restart_monitor)
        self.btn_refresh_devices = QPushButton("↻")
        self.btn_refresh_devices.setToolTip("Refresh device list")
        self.btn_refresh_devices.setFixedWidth(30)
        self.btn_refresh_devices.clicked.connect(self._refresh_devices)
        dev_row.addWidget(self.combo_devices, 1)
        dev_row.addWidget(self.btn_refresh_devices)
        layout.addLayout(dev_row)
        self._refresh_devices()

        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Digital Gain:"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal);
        self.slider_gain.setRange(10, 100);
        self.slider_gain.setValue(10)
        self.lbl_gain_val = QLabel("1.0x")
        self.slider_gain.valueChanged.connect(
            lambda v: (self.lbl_gain_val.setText(f"{v / 10:.1f}x"), self._update_runtime_config(),
                       self._update_monitor_gain()))
        gain_layout.addWidget(self.slider_gain);
        gain_layout.addWidget(self.lbl_gain_val)
        layout.addLayout(gain_layout)

        # --- TRAIN SECTION ---
        train_group = QGroupBox("Training")
        t_layout = QVBoxLayout(train_group)
        self.lbl_pattern_count = QLabel("Patterns Saved: 0")
        t_layout.addWidget(self.lbl_pattern_count)

        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton(f"TRAIN MODE ({self.txt_train})")
        self.btn_train.setStyleSheet(self.STYLE_ORANGE)
        self.btn_train.clicked.connect(self._toggle_train_mode)

        self.btn_train_key = QPushButton("Set Key")
        self.btn_train_key.setToolTip("Rebind the TRAIN hotkey")
        self.btn_train_key.clicked.connect(lambda: self._wait_key('train'))

        self.btn_reset_train = QPushButton("Reset Patterns")
        self.btn_reset_train.clicked.connect(self._reset_patterns)

        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_train_key)
        btn_layout.addWidget(self.btn_reset_train)
        t_layout.addLayout(btn_layout)
        t_layout.addWidget(
            QLabel("1. Enable Train Mode.\n2. Cast manually (Click 1 - IGNORED).\n3. Catch fish (Click 2 - SAVED)."))
        layout.addWidget(train_group)

        layout.addWidget(QLabel("1. Raw Volume:"));
        self.volume_bar = QProgressBar();
        self.volume_bar.setRange(0, 100);
        self.volume_bar.setFormat("%v %")
        self.volume_bar.setStyleSheet("QProgressBar::chunk { background-color: #2196F3; }");
        layout.addWidget(self.volume_bar)
        layout.addWidget(QLabel("2. Max Match Score:"));
        self.audio_bar = QProgressBar();
        self.audio_bar.setRange(0, 100);
        self.audio_bar.setFormat("%v %")
        self.audio_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }");
        layout.addWidget(self.audio_bar)
        self.lbl_audio_state = QLabel("Status: Ready");
        layout.addWidget(self.lbl_audio_state)

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
    #           LOGIC
    # ==========================================

    def _update_pattern_count(self):
        count = 1 if os.path.exists(resource_path('Splash_1.wav')) else 0
        count += len(glob.glob(os.path.join(get_app_data_dir(), "Splash_*.wav")))
        self.lbl_pattern_count.setText(f"Patterns Available: {count}")

    def _reset_patterns(self):
        if QMessageBox.question(self, "Reset", "Delete all temp patterns?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            cleanup_temp_patterns()
            self._update_pattern_count()

    def _toggle_train_mode(self):
        if self.train_worker:
            self._stop_train_mode("Stopped")
        else:
            self._stop_monitor()
            self.tabs.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.btn_train.setText("LISTENING (Cast then Catch)")
            self.btn_train.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")

            dev_idx = self.combo_devices.currentData()
            gain = self.slider_gain.value() / 10.0

            self.train_worker = AudioTrainerWorker(dev_idx, gain)
            self.train_worker.volume_level.connect(self.volume_bar.setValue)
            self.train_worker.pattern_saved.connect(
                lambda c: (self._update_pattern_count(), self.lbl_audio_state.setText(f"Captured Pattern")))
            self.train_worker.training_log.connect(lambda s: self.lbl_audio_state.setText(s))
            self.train_worker.start()

    def _stop_train_mode(self, msg):
        if self.train_worker:
            self.train_worker.stop()
            self.train_worker = None

        self.tabs.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_train.setText(f"TRAIN MODE ({self.txt_train})")
        self.btn_train.setStyleSheet(self.STYLE_ORANGE)
        self._start_monitor()
        self._update_pattern_count()

    # --- Standard Worker Logic ---
    def _get_current_ui_config(self):
        return {
            'x': self.spin_x.value(), 'y': self.spin_y.value(),
            'threshold': self.spin_th.value(), 'sensitivity': self.spin_sens.value(),
            'visual_timeout': self.spin_timeout.value(),
            'audio_threshold': self.slider_audio_th.value(),
            'audio_gain': self.slider_gain.value() / 10.0,
            'use_potions': self.chk_pot.isChecked(), 'potion_delay': self.spin_delay.value(),
            'potion_vk': self.vk_potion,
        }

    def _update_runtime_config(self):
        # LOCK: Don't send config if we are in the middle of loading a profile
        if self.loading_profile: return
        if self.active_worker: self.active_worker.update_config(self._get_current_ui_config())

    def _toggle_active_worker(self):
        if self.active_worker:
            self.active_worker.stop();
            self.active_worker = None
            self.btn_start.setText(f"Start Fishing ({self.txt_fishing})");
            self.btn_start.setStyleSheet(self.STYLE_GREEN)
            self.tabs.setEnabled(True);
            self.lbl_vis_state.setText("State: STOPPED");
            self.lbl_audio_state.setText("Status: Stopped")
            self._start_monitor()
        else:
            self._stop_monitor();
            self.tabs.setEnabled(False)
            cfg = self._get_current_ui_config()
            if self.tabs.currentIndex() == 0:
                if int(cfg['x']) == 0 and int(cfg['y']) == 0:
                    QMessageBox.warning(self, "Set Position First",
                                        f"Hover over your bobber and press {self.txt_pos} (or click 'Set Pos Key') "
                                        f"to set the X/Y target before starting Visual Mode.")
                    self.tabs.setEnabled(True);
                    self._start_monitor()
                    return
                self.active_worker = VisualWorker(cfg)
                self.active_worker.image_processed.connect(lambda r, p: (self.lbl_raw.setPixmap(QPixmap.fromImage(r)),
                                                                         self.lbl_proc.setPixmap(QPixmap.fromImage(p))))
                self.active_worker.stats_updated.connect(lambda s, t: (self.lbl_vis_state.setText(f"State: {t}"),
                                                                       self.vis_prog_bar.setValue(
                                                                           min(100, int(s * 100)))))
            else:
                dev_idx = self.combo_devices.currentData()
                if dev_idx is None:
                    QMessageBox.warning(self, "No Device",
                                        "No audio input device selected. Install VB-CABLE and click ↻.")
                    self.tabs.setEnabled(True);
                    self._start_monitor()
                    return
                self.active_worker = AudioPatternWorker(cfg, dev_idx)
                self.active_worker.match_score.connect(self.audio_bar.setValue)
                self.active_worker.current_volume.connect(self.volume_bar.setValue)
                self.active_worker.state_updated.connect(self.lbl_audio_state.setText)
            self.active_worker.potion_drank.connect(self._on_potion_update)
            self.active_worker.start()
            self.btn_start.setText(f"Stop Fishing ({self.txt_fishing})");
            self.btn_start.setStyleSheet(self.STYLE_RED)

    def _on_potion_update(self, t):
        self.lbl_potion.setText(f"Drink in: {t}s" if t > 0 else ("DRANK!" if t == 0 else "OFF"))

    def _start_monitor(self):
        if self.monitor_worker: return
        dev_idx = self.combo_devices.currentData()
        gain = self.slider_gain.value() / 10.0
        self.monitor_worker = AudioMonitorWorker(dev_idx, gain)
        self.monitor_worker.current_volume.connect(self.volume_bar.setValue)
        self.monitor_worker.start()

    def _stop_monitor(self):
        if self.monitor_worker: self.monitor_worker.stop(); self.monitor_worker = None; self.volume_bar.setValue(0)

    def _restart_monitor(self):
        if not self.active_worker: self._stop_monitor(); self._start_monitor()

    def _refresh_devices(self):
        prev_idx = self.combo_devices.currentData()
        was_monitoring = self.monitor_worker is not None
        if was_monitoring: self._stop_monitor()

        self.combo_devices.blockSignals(True)
        self.combo_devices.clear()
        found = 0
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    self.combo_devices.addItem(f"{idx}: {dev['name']}", idx)
                    found += 1
        except Exception as e:
            logging.warning(f"sd.query_devices failed: {e}")

        if found == 0:
            self.combo_devices.addItem("(no input devices found)", None)
            if hasattr(self, 'lbl_audio_state'):
                self.lbl_audio_state.setText("Status: No audio input devices. Install VB-CABLE or check drivers.")
        else:
            # Restore previous selection if still present
            for i in range(self.combo_devices.count()):
                if self.combo_devices.itemData(i) == prev_idx:
                    self.combo_devices.setCurrentIndex(i)
                    break

        self.combo_devices.blockSignals(False)
        if was_monitoring and found > 0: self._start_monitor()

    def _update_monitor_gain(self):
        if self.monitor_worker: self.monitor_worker.update_settings(self.combo_devices.currentData(),
                                                                    self.slider_gain.value() / 10.0)

    # --- PROFILES ---
    def _load_profiles_from_file(self):
        self.config.read(self.CONFIG_FILE)
        if not self.config.sections():
            self.config['Default'] = {'x': '0', 'y': '0', 'vis_th': '10', 'vis_sens': '50', 'potions': 'False',
                                      'pot_delay': '180', 'audio_th': '60', 'audio_gain': '1.0',
                                      'potion_vk': '66'}
            with open(self.CONFIG_FILE, 'w') as f: self.config.write(f)
        self.combo_profiles.blockSignals(True);
        self.combo_profiles.clear()
        for s in self.config.sections(): self.combo_profiles.addItem(s)
        self.combo_profiles.blockSignals(False);
        self._on_profile_changed()

    @staticmethod
    def _safe_int(d, key, default):
        try:
            return int(d.get(key, default))
        except (ValueError, TypeError):
            logging.warning(f"Profile field '{key}'={d.get(key)!r} is not an int; using {default}")
            return default

    @staticmethod
    def _safe_float(d, key, default):
        try:
            return float(d.get(key, default))
        except (ValueError, TypeError):
            logging.warning(f"Profile field '{key}'={d.get(key)!r} is not a float; using {default}")
            return default

    def _on_profile_changed(self):
        self.loading_profile = True  # LOCK: Prevent update spam
        name = self.combo_profiles.currentText()
        if name in self.config:
            d = self.config[name]
            self.spin_x.setValue(self._safe_int(d, 'x', 0));
            self.spin_y.setValue(self._safe_int(d, 'y', 0))
            self.spin_th.setValue(self._safe_int(d, 'vis_th', 10));
            self.spin_sens.setValue(self._safe_int(d, 'vis_sens', 50))
            self.spin_timeout.setValue(self._safe_int(d, 'vis_timeout', 10))
            self.slider_audio_th.setValue(self._safe_int(d, 'audio_th', 60));
            self.slider_gain.setValue(int(self._safe_float(d, 'audio_gain', 1.0) * 10))
            self.chk_pot.setChecked(d.get('potions', 'False') == 'True');
            self.spin_delay.setValue(self._safe_int(d, 'pot_delay', 180))
            self.vk_potion = self._safe_int(d, 'potion_vk', 66)
            try:
                ch = KeyCode.from_vk(self.vk_potion).char
                self.txt_potion = ch.upper() if ch else f"VK_{self.vk_potion}"
            except Exception:
                self.txt_potion = f"VK_{self.vk_potion}"
            self.btn_potion_key.setText(f"Key: {self.txt_potion}")

        self.loading_profile = False  # UNLOCK
        self._update_runtime_config()  # Send ONE final update

    def _save_current_profile(self):
        n = self.combo_profiles.currentText()
        if n:
            self.config[n] = {'x': str(self.spin_x.value()), 'y': str(self.spin_y.value()),
                              'vis_th': str(self.spin_th.value()), 'vis_sens': str(self.spin_sens.value()),
                              'audio_th': str(self.slider_audio_th.value()),
                              'audio_gain': str(self.slider_gain.value() / 10.0),
                              'potions': str(self.chk_pot.isChecked()), 'pot_delay': str(self.spin_delay.value()),
                              'potion_vk': str(self.vk_potion),
                              'vis_timeout': str(self.spin_timeout.value())}
            with open(self.CONFIG_FILE, 'w') as f: self.config.write(f)
            QMessageBox.information(self, "Saved", f"Profile '{n}' saved!")

    def _new_profile(self):
        t, o = QInputDialog.getText(self, 'New', 'Name:');
        if o and t: self.config[t] = {}; self.combo_profiles.addItem(t); self.combo_profiles.setCurrentIndex(
            self.combo_profiles.count() - 1); self._save_current_profile()

    def _del_profile(self):
        n = self.combo_profiles.currentText()
        if self.combo_profiles.count() > 1 and QMessageBox.question(self, "Del", f"Del '{n}'?",
                                                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.config.remove_section(n);
            with open(self.CONFIG_FILE, 'w') as f: self.config.write(f)
            self._load_profiles_from_file()

    # --- HOTKEYS ---
    def _toggle_hotkeys_state(self):
        self.hotkeys_active = not self.hotkeys_active; self._refresh_hotkey_btn()

    def _refresh_hotkey_btn(self):
        self.btn_hotkeys.setText("HOTKEYS: ON" if self.hotkeys_active else "HOTKEYS: MUTED")
        self.btn_hotkeys.setStyleSheet(self.STYLE_RED if self.hotkeys_active else self.STYLE_GREEN)

    def _handle_hotkey_vk(self, vk):
        if not self.hotkeys_active or self.rebinding_key:
            return
        if vk == self.vk_fishing:
            if self.btn_start.isEnabled():
                self._toggle_active_worker()
        elif vk == self.vk_train:
            self._toggle_train_mode()
        elif vk == self.vk_pos:
            pos = pyautogui.position(); self.spin_x.setValue(pos.x); self.spin_y.setValue(pos.y)

    def _wait_key(self, target):
        d = QDialog(self);
        d.setWindowTitle("Press Key");
        d.setFixedSize(200, 100)
        l = QLabel("Press any key...", d);
        l.setAlignment(Qt.AlignmentFlag.AlignCenter);
        layout = QVBoxLayout(d);
        layout.addWidget(l)

        # The pynput listener runs on its own OS thread. Do NOT touch Qt widgets
        # from on_press — just capture the key and ask the main thread to close
        # the dialog. All widget updates happen after d.exec() returns.
        captured = {'vk': None, 'char': None}

        def on_press(key):
            try:
                vk = key.vk if hasattr(key, 'vk') else key.value.vk
            except AttributeError:
                vk = None
            char = key.char.upper() if hasattr(key, 'char') and key.char else (f"VK_{vk}" if vk else "?")
            if vk:
                captured['vk'] = vk
                captured['char'] = char
                QTimer.singleShot(0, d.accept)
                return False

        self.rebinding_key = True
        try:
            listener = Listener(on_press=on_press);
            listener.start();
            d.exec();
            listener.stop()
        finally:
            self.rebinding_key = False

        vk, char = captured['vk'], captured['char']
        if not vk: return
        if target == 'main':
            self.vk_fishing = vk; self.txt_fishing = char; self.btn_start.setText(f"Start Fishing ({char})")
        elif target == 'pos':
            self.vk_pos = vk; self.txt_pos = char; self.btn_pos_key.setText(f"Set Pos Key ({char})")
        elif target == 'potion':
            self.vk_potion = vk; self.txt_potion = char; self.btn_potion_key.setText(f"Key: {char}")
            self._update_runtime_config()
        elif target == 'train':
            self.vk_train = vk; self.txt_train = char
            self.btn_train.setText(f"TRAIN MODE ({char})")


def main():
    cleanup_temp_patterns()
    app = QApplication(sys.argv)
    window = AppUi()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()