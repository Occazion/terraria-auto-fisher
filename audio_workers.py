"""Three audio workers backing the Audio Mode tab.

* AudioMonitorWorker  — live volume meter for the UI.
* AudioTrainerWorker  — records mouse-click-bracketed splashes to disk.
* AudioPatternWorker  — runs FFT cross-correlation against saved patterns
                        and triggers the catch-then-cast click sequence.
"""

import collections
import glob
import logging
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import QThread, pyqtSignal
from pynput import mouse
from scipy import signal

from input_control import MouseController, tick_potion
from paths import get_app_data_dir, resource_path


# Audio pipeline constants
_DEFAULT_SAMPLERATE = 44100
_PROCESS_SAMPLERATE = 8000        # downsample target for correlation
_HIGHPASS_HZ = 1000               # filter out background rumble before matching
_HIGHPASS_ORDER = 10
_RMS_WINDOW = 100                 # samples per RMS envelope window
_MIN_VOLUME_PEAK = 0.02           # below this, skip matching (silence)
_MIN_LIVE_ENVELOPE = 0.001        # numerical-stability floor
_SCORE_SCALE = 1.2                # empirical multiplier on (peak / template_len) * 100

# Trainer constants
_TRAINER_BUFFER_CHUNKS = 40       # ring buffer of recent audio chunks
_TRAINER_SAVE_DURATION_SEC = 1.5  # how much trailing audio to keep around the splash
_TRAINER_PRE_PEAK_SEC = 0.15      # window around the peak when trimming
_TRAINER_POST_PEAK_SEC = 0.35
_TRAINER_BUNDLED_INDEX = 1        # Splash_1.wav is the bundled pattern; saves start at 2

# Trigger constants
_TRIGGER_REEL_WAIT_SEC = 2.5      # time between reel-in click and re-cast click
_TRIGGER_RECAST_WAIT_SEC = 1.5    # time after re-cast before resuming detection
_TRIGGER_COOLDOWN_PAD_SEC = 0.5   # extra suppression past the action duration


class AudioMonitorWorker(QThread):
    """Lightweight volume meter — just emits a 0-100 value per audio block."""

    current_volume = pyqtSignal(int)

    def __init__(self, device_idx, gain: float = 1.0):
        super().__init__()
        self.running = True
        self.device_idx = device_idx
        self.gain = gain

    def update_settings(self, device_idx, gain: float) -> None:
        self.device_idx = device_idx
        self.gain = gain

    def run(self):
        def callback(indata, frames, time_info, status):
            if not self.running:
                raise sd.CallbackStop()
            vol = np.max(np.abs(indata)) * self.gain
            self.current_volume.emit(int(min(100, vol * 100)))

        try:
            with sd.InputStream(device=self.device_idx, channels=1,
                                callback=callback, blocksize=4096):
                while self.running:
                    sd.sleep(100)
        except Exception:
            self.current_volume.emit(0)

    def stop(self) -> None:
        self.running = False
        self.wait()


class AudioTrainerWorker(QThread):
    """Listens for mouse clicks; on every even-numbered click saves the
    trailing audio buffer trimmed around its loudest peak."""

    pattern_saved = pyqtSignal(int)
    training_log = pyqtSignal(str)
    volume_level = pyqtSignal(int)

    def __init__(self, device_idx, gain: float = 1.0):
        super().__init__()
        self.running = True
        self.device_idx = device_idx
        self.gain = gain
        self.buffer = collections.deque(maxlen=_TRAINER_BUFFER_CHUNKS)
        self.samplerate = _DEFAULT_SAMPLERATE
        self.mouse_listener = None
        self.click_count = 0

    def run(self):
        try:
            dev_info = sd.query_devices(self.device_idx, 'input')
            self.samplerate = int(dev_info['default_samplerate'])
        except Exception:
            pass

        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.mouse_listener.start()

        def callback(indata, frames, time_info, status):
            if not self.running:
                raise sd.CallbackStop()
            data = indata.flatten() * self.gain
            self.buffer.append(data.copy())
            vol = np.max(np.abs(data))
            self.volume_level.emit(int(min(100, vol * 100)))

        try:
            with sd.InputStream(device=self.device_idx, channels=1,
                                callback=callback, samplerate=self.samplerate):
                while self.running:
                    sd.sleep(50)
        except Exception as e:
            self.training_log.emit(f"Error: {e}")

    def _on_click(self, x, y, button, pressed):
        if not self.running:
            return False
        if pressed and button == mouse.Button.left:
            self.click_count += 1
            if self.click_count % 2 != 0:
                self.training_log.emit("Casted... WAIT FOR SPLASH!")
                self.buffer.clear()
            else:
                self._save_next_pattern()
                self.training_log.emit("Caught! Cast again...")

    def _save_next_pattern(self) -> None:
        try:
            app_data_path = get_app_data_dir()
            existing = glob.glob(os.path.join(app_data_path, "Splash_*.wav"))
            # Pick max(index)+1 so non-contiguous files (post-delete) don't collide.
            used = []
            for p in existing:
                try:
                    used.append(int(os.path.splitext(os.path.basename(p))[0].split('_')[1]))
                except (ValueError, IndexError):
                    pass
            next_idx = max(used + [_TRAINER_BUNDLED_INDEX]) + 1
            filename = os.path.join(app_data_path, f"Splash_{next_idx}.wav")

            full_audio = np.concatenate(list(self.buffer))
            duration_samples = int(_TRAINER_SAVE_DURATION_SEC * self.samplerate)
            if len(full_audio) > duration_samples:
                full_audio = full_audio[-duration_samples:]

            trimmed = self._smart_trim(full_audio)
            sf.write(filename, trimmed, self.samplerate)
            self.pattern_saved.emit(next_idx)
        except Exception as e:
            self.training_log.emit(f"Save Error: {e}")

    def _smart_trim(self, data):
        peak_idx = np.argmax(np.abs(data))
        pre = int(_TRAINER_PRE_PEAK_SEC * self.samplerate)
        post = int(_TRAINER_POST_PEAK_SEC * self.samplerate)
        start = max(0, peak_idx - pre)
        end = min(len(data), peak_idx + post)
        return data[start:end]

    def stop(self) -> None:
        self.running = False
        if self.mouse_listener:
            self.mouse_listener.stop()
        self.wait()


class AudioPatternWorker(QThread):
    """FFT cross-correlation against recorded splash patterns.

    On a match the PortAudio callback only flips a flag + starts a cooldown;
    the actual click sequence runs on the worker thread (see _do_catch_sequence)
    so we never block inside an audio callback.
    """

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
        self.device_rate = _DEFAULT_SAMPLERATE
        self.process_rate = _PROCESS_SAMPLERATE
        self.sos_filter = None  # built in _prepare_templates

        self._trigger_pending = False
        self._cooldown_until = 0.0
        self._trigger_action_sec = _TRIGGER_REEL_WAIT_SEC + _TRIGGER_RECAST_WAIT_SEC

    # ------------- setup (runs on worker thread, not main) -------------

    def _prepare_templates(self) -> None:
        try:
            try:
                dev_info = sd.query_devices(self.device_idx, 'input')
                self.device_rate = int(dev_info['default_samplerate'])
            except Exception:
                self.device_rate = _DEFAULT_SAMPLERATE

            bundled_path = resource_path('Splash_1.wav')
            temp_files = glob.glob(os.path.join(get_app_data_dir(), "Splash_*.wav"))

            all_files = []
            if os.path.exists(bundled_path):
                all_files.append(bundled_path)
            all_files.extend(temp_files)

            if not all_files:
                self.state_updated.emit("Error: No Splash_1.wav found!")
                self.running = False
                return

            self.sos_filter = signal.butter(_HIGHPASS_ORDER, _HIGHPASS_HZ, 'hp',
                                            fs=self.process_rate, output='sos')

            count = 0
            for p_file in all_files:
                try:
                    self.templates.append(self._build_template(p_file))
                    count += 1
                except Exception as e:
                    logging.warning(f"Skipping pattern {p_file}: {e}")

            self.state_updated.emit(f"Ready ({count} patterns)")
        except Exception as e:
            self.state_updated.emit(f"Init Error: {e}")
            self.running = False

    def _build_template(self, path: str):
        data, file_rate = sf.read(path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        data = self._smart_trim(data, file_rate)
        target_len = int(len(data) * self.process_rate / file_rate)
        data = signal.resample(data, target_len)
        filtered = signal.sosfilt(self.sos_filter, data)
        env = self._get_rms_envelope(filtered)
        env = env - np.min(env)
        max_val = np.max(env)
        if max_val > 0:
            env = env / max_val
        return env

    @staticmethod
    def _smart_trim(data, rate, duration_sec: float = 0.4):
        if len(data) <= rate * duration_sec:
            return data
        peak_idx = np.argmax(np.abs(data))
        half_window = int((rate * duration_sec) / 2)
        start = max(0, peak_idx - half_window)
        end = min(len(data), peak_idx + half_window)
        return data[start:end]

    @staticmethod
    def _get_rms_envelope(data, window_size: int = _RMS_WINDOW):
        squared = np.power(data, 2)
        window = np.ones(window_size) / window_size
        mean_squared = np.convolve(squared, window, mode='same')
        return np.sqrt(mean_squared)

    def update_config(self, new_config) -> None:
        self.cfg = new_config

    # ------------- main loop -------------

    def run(self):
        self._prepare_templates()
        if not self.running or not self.templates:
            return

        max_len = max(len(t) for t in self.templates)
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

                live_audio = indata.flatten() * float(self.cfg['audio_gain'])

                vol_peak = np.max(np.abs(live_audio))
                if not self.is_stopping:
                    self.current_volume.emit(int(min(100, vol_peak * 100)))

                if vol_peak < _MIN_VOLUME_PEAK:
                    if not self.is_stopping:
                        self.match_score.emit(0)
                    return

                live_downsampled = live_audio[::downsample_factor]
                live_filtered = signal.sosfilt(self.sos_filter, live_downsampled)
                live_envelope = self._get_rms_envelope(live_filtered)
                live_envelope = live_envelope - np.min(live_envelope)
                live_max = np.max(live_envelope)
                if live_max > _MIN_LIVE_ENVELOPE:
                    live_envelope = live_envelope / live_max

                best_score = 0
                for temp in self.templates:
                    correlation = signal.correlate(live_envelope, temp, mode='valid', method='fft')
                    if correlation.size == 0:
                        correlation = signal.correlate(live_envelope, temp, mode='same', method='fft')
                    peak = np.max(correlation)
                    score = int(min(100, (peak / len(temp)) * 100 * _SCORE_SCALE))
                    if score > best_score:
                        best_score = score

                if self.is_stopping:
                    return
                self.match_score.emit(best_score)
                if best_score > int(self.cfg['audio_threshold']):
                    # Hand off to run() — never block inside a PortAudio callback.
                    self._cooldown_until = time.time() + self._trigger_action_sec + _TRIGGER_COOLDOWN_PAD_SEC
                    self._trigger_pending = True
                    self.trigger_fired.emit("MATCH!")
                    self.state_updated.emit(f"SPLASH! ({best_score}%)")
            except Exception:
                pass

        try:
            with sd.InputStream(device=self.device_idx, channels=1, callback=callback,
                                blocksize=block_size, samplerate=self.device_rate):
                while self.running:
                    if self.is_stopping:
                        break
                    if self._trigger_pending:
                        self._trigger_pending = False
                        self._do_catch_sequence()
                    self._handle_potions()
                    sd.sleep(100)
        except Exception as e:
            if not self.is_stopping:
                self.state_updated.emit(f"Stream Error: {e}")

    def _do_catch_sequence(self) -> None:
        """Reel in, wait, re-cast. Runs on the worker thread."""
        MouseController.click()
        self._interruptible_sleep(_TRIGGER_REEL_WAIT_SEC)
        if self.is_stopping or not self.running:
            return
        MouseController.click()
        self._interruptible_sleep(_TRIGGER_RECAST_WAIT_SEC)
        if not self.is_stopping:
            self.state_updated.emit("Listening...")

    def _interruptible_sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            if self.is_stopping or not self.running:
                return
            time.sleep(0.05)

    def _handle_potions(self) -> None:
        self.potion_timer, signal_val = tick_potion(self.cfg, self.potion_timer)
        self.potion_drank.emit(signal_val)

    def stop(self) -> None:
        self.is_stopping = True
        self.running = False
        self.wait()
