"""AutoFisher main window.

AppUi orchestrates two tab widgets ([tab_visual.py](tab_visual.py),
[tab_audio.py](tab_audio.py)), profile persistence, the global hotkey
listener ([input_control.py](input_control.py)), and the worker thread
lifecycle ([visual_worker.py](visual_worker.py),
[audio_workers.py](audio_workers.py)).
"""

import configparser
import glob
import logging
import os

import pyautogui
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QGroupBox, QHBoxLayout, QInputDialog, QLabel,
                             QMainWindow, QMessageBox, QPushButton, QSpinBox,
                             QTabWidget, QVBoxLayout, QWidget)
from pynput.keyboard import Listener

from audio_workers import AudioMonitorWorker, AudioPatternWorker, AudioTrainerWorker
from input_control import KeyMonitor
from paths import cleanup_temp_patterns, get_app_data_dir, resource_path
from tab_audio import AudioTab
from tab_visual import VisualTab
from ui_common import (DEFAULT_VK_FISHING, DEFAULT_VK_POS, DEFAULT_VK_POTION,
                       DEFAULT_VK_TRAIN, STYLE_GREEN, STYLE_RED, safe_int,
                       vk_to_char)
from visual_worker import VisualWorker


CONFIG_FILE = 'fishing_profiles.ini'


class AppUi(QMainWindow):
    def __init__(self, version: str = ''):
        super().__init__()
        self.setWindowTitle(f'AutoFisher v{version}' if version else 'AutoFisher')
        self.resize(450, 680)

        self.config = configparser.ConfigParser()
        self.active_worker = None
        self.monitor_worker = None
        self.train_worker = None
        self.hotkeys_active = True
        self.loading_profile = False
        self.rebinding_key = False

        self.vk_fishing, self.txt_fishing = DEFAULT_VK_FISHING, 'F'
        self.vk_pos, self.txt_pos = DEFAULT_VK_POS, 'V'
        self.vk_train, self.txt_train = DEFAULT_VK_TRAIN, 'T'
        self.vk_potion, self.txt_potion = DEFAULT_VK_POTION, 'B'

        self._init_ui()
        self._wire_tab_signals()
        self._load_profiles_from_file()
        self._start_monitor()

        self.key_monitor = KeyMonitor()
        self.key_monitor.keyPressed.connect(self._handle_hotkey_vk)
        self._update_pattern_count()

    def closeEvent(self, event: QCloseEvent):
        self.hotkeys_active = False
        if self.key_monitor:
            try:
                self.key_monitor.stop()
            except Exception:
                pass
        for w in (self.active_worker, self.train_worker, self.monitor_worker):
            if w is None:
                continue
            try:
                w.stop()
            except Exception as e:
                logging.warning(f"Worker stop failed: {e}")
        event.accept()

    # ==========================================
    #              UI CONSTRUCTION
    # ==========================================

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(self._build_profile_group())

        self.btn_hotkeys = QPushButton("HOTKEYS: ON")
        self.btn_hotkeys.clicked.connect(self._toggle_hotkeys_state)
        self._refresh_hotkey_btn()
        layout.addWidget(self.btn_hotkeys)

        self.tabs = QTabWidget()
        self.visual_tab = VisualTab(pos_key_label=self.txt_pos)
        self.audio_tab = AudioTab(train_key_label=self.txt_train)
        self.tabs.addTab(self.visual_tab, "Visual Mode")
        self.tabs.addTab(self.audio_tab, "Audio Pattern Mode")
        layout.addWidget(self.tabs)

        layout.addLayout(self._build_footer())

    def _build_profile_group(self) -> QGroupBox:
        group = QGroupBox("Profile Management")
        v = QVBoxLayout(group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Profile:"))

        self.combo_profiles = QComboBox()
        self.combo_profiles.currentIndexChanged.connect(self._on_profile_changed)
        row.addWidget(self.combo_profiles, 1)

        for label, slot in (("New", self._new_profile),
                            ("Del", self._del_profile),
                            ("Save", self._save_current_profile)):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            row.addWidget(btn)

        v.addLayout(row)
        return group

    def _build_footer(self) -> QVBoxLayout:
        footer = QVBoxLayout()

        # Potion row
        pot_row = QHBoxLayout()
        self.chk_pot = QCheckBox("Drink Potions")
        self.chk_pot.stateChanged.connect(self._on_config_changed)
        self.spin_delay = QSpinBox()
        self.spin_delay.setRange(0, 3600)
        self.spin_delay.setSuffix("s")
        self.spin_delay.valueChanged.connect(self._on_config_changed)
        self.lbl_potion = QLabel("Potions: OFF")
        self.btn_potion_key = QPushButton(f"Key: {self.txt_potion}")
        self.btn_potion_key.setToolTip("Key sent to drink potion")
        self.btn_potion_key.clicked.connect(lambda: self._wait_key('potion'))
        for w in (self.chk_pot, self.spin_delay, self.btn_potion_key, self.lbl_potion):
            pot_row.addWidget(w)
        footer.addLayout(pot_row)

        # Start button
        self.btn_start = QPushButton(f"Start Fishing ({self.txt_fishing})")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet(STYLE_GREEN)
        self.btn_start.clicked.connect(self._toggle_active_worker)
        footer.addWidget(self.btn_start)

        # Rebind start hotkey
        hk_row = QHBoxLayout()
        self.btn_main_key = QPushButton("Change Start Key")
        self.btn_main_key.clicked.connect(lambda: self._wait_key('main'))
        hk_row.addWidget(self.btn_main_key)
        footer.addLayout(hk_row)

        return footer

    def _wire_tab_signals(self) -> None:
        # Visual tab
        self.visual_tab.config_changed.connect(self._on_config_changed)
        self.visual_tab.rebind_pos_requested.connect(lambda: self._wait_key('pos'))

        # Audio tab
        self.audio_tab.config_changed.connect(self._on_config_changed)
        self.audio_tab.gain_changed.connect(self._update_monitor_gain)
        self.audio_tab.device_changed.connect(self._restart_monitor)
        self.audio_tab.refresh_devices_requested.connect(self._refresh_devices)
        self.audio_tab.toggle_train_requested.connect(self._toggle_train_mode)
        self.audio_tab.rebind_train_requested.connect(lambda: self._wait_key('train'))
        self.audio_tab.reset_patterns_requested.connect(self._reset_patterns)

        # Initial device population happens here (after signals are wired, but
        # device_changed is harmless since active_worker is None).
        self.audio_tab.populate_devices()

    # ==========================================
    #         PATTERNS / TRAINING
    # ==========================================

    def _update_pattern_count(self) -> None:
        count = 1 if os.path.exists(resource_path('Splash_1.wav')) else 0
        count += len(glob.glob(os.path.join(get_app_data_dir(), "Splash_*.wav")))
        self.audio_tab.update_pattern_count(count)

    def _reset_patterns(self) -> None:
        confirm = QMessageBox.question(
            self, "Reset", "Delete all temp patterns?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            cleanup_temp_patterns()
            self._update_pattern_count()

    def _toggle_train_mode(self) -> None:
        if self.train_worker:
            self._stop_train_mode()
            return

        self._stop_monitor()
        self.tabs.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.audio_tab.set_train_listening()

        dev_idx = self.audio_tab.get_selected_device()
        gain = self.audio_tab.get_gain()
        self.train_worker = AudioTrainerWorker(dev_idx, gain)
        self.train_worker.volume_level.connect(self.audio_tab.update_volume)
        self.train_worker.pattern_saved.connect(self._on_pattern_saved)
        self.train_worker.training_log.connect(self.audio_tab.set_state_text)
        self.train_worker.start()

    def _on_pattern_saved(self, _count: int) -> None:
        self._update_pattern_count()
        self.audio_tab.set_state_text("Captured Pattern")

    def _stop_train_mode(self) -> None:
        if self.train_worker:
            self.train_worker.stop()
            self.train_worker = None
        self.tabs.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.audio_tab.set_train_idle()
        self._start_monitor()
        self._update_pattern_count()

    # ==========================================
    #         WORKER LIFECYCLE
    # ==========================================

    def _get_current_ui_config(self) -> dict:
        cfg = {}
        cfg.update(self.visual_tab.to_runtime_config())
        cfg.update(self.audio_tab.to_runtime_config())
        cfg.update({
            'use_potions': self.chk_pot.isChecked(),
            'potion_delay': self.spin_delay.value(),
            'potion_vk': self.vk_potion,
        })
        return cfg

    def _on_config_changed(self) -> None:
        if self.loading_profile:
            return
        if self.active_worker:
            self.active_worker.update_config(self._get_current_ui_config())

    def _toggle_active_worker(self) -> None:
        if self.active_worker:
            self._stop_active_worker()
        else:
            self._start_active_worker()

    def _stop_active_worker(self) -> None:
        self.active_worker.stop()
        self.active_worker = None
        self.btn_start.setText(f"Start Fishing ({self.txt_fishing})")
        self.btn_start.setStyleSheet(STYLE_GREEN)
        self.tabs.setEnabled(True)
        self.visual_tab.set_state_text("State: STOPPED")
        self.audio_tab.set_state_text("Status: Stopped")
        self._start_monitor()

    def _start_active_worker(self) -> None:
        self._stop_monitor()
        self.tabs.setEnabled(False)
        cfg = self._get_current_ui_config()

        if self.tabs.currentIndex() == 0:
            if not self._start_visual_worker(cfg):
                return
        else:
            if not self._start_audio_worker(cfg):
                return

        self.active_worker.potion_drank.connect(self._on_potion_update)
        self.active_worker.start()
        self.btn_start.setText(f"Stop Fishing ({self.txt_fishing})")
        self.btn_start.setStyleSheet(STYLE_RED)

    def _start_visual_worker(self, cfg: dict) -> bool:
        if not self.visual_tab.has_position():
            QMessageBox.warning(
                self, "Set Position First",
                f"Hover over your bobber and press {self.txt_pos} (or click 'Set Pos Key') "
                f"to set the X/Y target before starting Visual Mode."
            )
            self.tabs.setEnabled(True)
            self._start_monitor()
            return False
        self.active_worker = VisualWorker(cfg)
        self.active_worker.image_processed.connect(self.visual_tab.update_frame)
        self.active_worker.stats_updated.connect(self.visual_tab.update_stats)
        return True

    def _start_audio_worker(self, cfg: dict) -> bool:
        dev_idx = self.audio_tab.get_selected_device()
        if dev_idx is None:
            QMessageBox.warning(
                self, "No Device",
                "No audio input device selected. Install VB-CABLE and click ↻."
            )
            self.tabs.setEnabled(True)
            self._start_monitor()
            return False
        self.active_worker = AudioPatternWorker(cfg, dev_idx)
        self.active_worker.match_score.connect(self.audio_tab.update_score)
        self.active_worker.current_volume.connect(self.audio_tab.update_volume)
        self.active_worker.state_updated.connect(self.audio_tab.set_state_text)
        return True

    def _on_potion_update(self, t: int) -> None:
        if t > 0:
            self.lbl_potion.setText(f"Drink in: {t}s")
        elif t == 0:
            self.lbl_potion.setText("DRANK!")
        else:
            self.lbl_potion.setText("OFF")

    # ==========================================
    #         AUDIO DEVICE MONITOR
    # ==========================================

    def _start_monitor(self) -> None:
        if self.monitor_worker:
            return
        dev_idx = self.audio_tab.get_selected_device()
        gain = self.audio_tab.get_gain()
        self.monitor_worker = AudioMonitorWorker(dev_idx, gain)
        self.monitor_worker.current_volume.connect(self.audio_tab.update_volume)
        self.monitor_worker.start()

    def _stop_monitor(self) -> None:
        if not self.monitor_worker:
            return
        self.monitor_worker.stop()
        self.monitor_worker = None
        self.audio_tab.reset_volume()

    def _restart_monitor(self) -> None:
        if self.active_worker:
            return
        self._stop_monitor()
        self._start_monitor()

    def _refresh_devices(self) -> None:
        was_monitoring = self.monitor_worker is not None
        if was_monitoring:
            self._stop_monitor()
        found = self.audio_tab.populate_devices()
        if was_monitoring and found > 0:
            self._start_monitor()

    def _update_monitor_gain(self) -> None:
        if self.monitor_worker:
            self.monitor_worker.update_settings(
                self.audio_tab.get_selected_device(),
                self.audio_tab.get_gain(),
            )

    # ==========================================
    #              PROFILES
    # ==========================================

    def _load_profiles_from_file(self) -> None:
        self.config.read(CONFIG_FILE)
        if not self.config.sections():
            self.config['Default'] = {
                'x': '0', 'y': '0',
                'vis_th': '10', 'vis_sens': '50', 'vis_timeout': '10',
                'potions': 'False', 'pot_delay': '180', 'potion_vk': '66',
                'audio_th': '60', 'audio_gain': '1.0',
            }
            with open(CONFIG_FILE, 'w') as f:
                self.config.write(f)
        self.combo_profiles.blockSignals(True)
        self.combo_profiles.clear()
        for s in self.config.sections():
            self.combo_profiles.addItem(s)
        self.combo_profiles.blockSignals(False)
        self._on_profile_changed()

    def _on_profile_changed(self) -> None:
        self.loading_profile = True
        try:
            name = self.combo_profiles.currentText()
            if name in self.config:
                d = self.config[name]
                self.visual_tab.from_profile_dict(d)
                self.audio_tab.from_profile_dict(d)
                self.chk_pot.setChecked(d.get('potions', 'False') == 'True')
                self.spin_delay.setValue(safe_int(d, 'pot_delay', 180))
                self.vk_potion = safe_int(d, 'potion_vk', DEFAULT_VK_POTION)
                self.txt_potion = vk_to_char(self.vk_potion)
                self.btn_potion_key.setText(f"Key: {self.txt_potion}")
        finally:
            self.loading_profile = False
        self._on_config_changed()

    def _save_current_profile(self) -> None:
        n = self.combo_profiles.currentText()
        if not n:
            return
        section = {}
        section.update(self.visual_tab.to_profile_dict())
        section.update(self.audio_tab.to_profile_dict())
        section.update({
            'potions': str(self.chk_pot.isChecked()),
            'pot_delay': str(self.spin_delay.value()),
            'potion_vk': str(self.vk_potion),
        })
        self.config[n] = section
        with open(CONFIG_FILE, 'w') as f:
            self.config.write(f)
        QMessageBox.information(self, "Saved", f"Profile '{n}' saved!")

    def _new_profile(self) -> None:
        name, ok = QInputDialog.getText(self, 'New', 'Name:')
        if not (ok and name):
            return
        self.config[name] = {}
        self.combo_profiles.addItem(name)
        self.combo_profiles.setCurrentIndex(self.combo_profiles.count() - 1)
        self._save_current_profile()

    def _del_profile(self) -> None:
        n = self.combo_profiles.currentText()
        if self.combo_profiles.count() <= 1:
            return
        confirm = QMessageBox.question(
            self, "Del", f"Del '{n}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.config.remove_section(n)
        with open(CONFIG_FILE, 'w') as f:
            self.config.write(f)
        self._load_profiles_from_file()

    # ==========================================
    #               HOTKEYS
    # ==========================================

    def _toggle_hotkeys_state(self) -> None:
        self.hotkeys_active = not self.hotkeys_active
        self._refresh_hotkey_btn()

    def _refresh_hotkey_btn(self) -> None:
        self.btn_hotkeys.setText("HOTKEYS: ON" if self.hotkeys_active else "HOTKEYS: MUTED")
        self.btn_hotkeys.setStyleSheet(STYLE_RED if self.hotkeys_active else STYLE_GREEN)

    def _handle_hotkey_vk(self, vk: int) -> None:
        if not self.hotkeys_active or self.rebinding_key:
            return
        if vk == self.vk_fishing:
            if self.btn_start.isEnabled():
                self._toggle_active_worker()
        elif vk == self.vk_train:
            self._toggle_train_mode()
        elif vk == self.vk_pos:
            pos = pyautogui.position()
            self.visual_tab.set_position(pos.x, pos.y)

    def _wait_key(self, target: str) -> None:
        """Pop a modal that captures the next keypress and rebinds `target`.

        The pynput listener runs on its own OS thread. Do NOT touch Qt widgets
        from on_press — capture the key, ask the main thread to close the
        dialog, then apply changes after d.exec() returns.
        """
        d = QDialog(self)
        d.setWindowTitle("Press Key")
        d.setFixedSize(200, 100)
        lbl = QLabel("Press any key...", d)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(d)
        layout.addWidget(lbl)

        captured = {'vk': None, 'char': None}

        def on_press(key):
            try:
                vk = key.vk if hasattr(key, 'vk') else key.value.vk
            except AttributeError:
                vk = None
            char = (key.char.upper() if hasattr(key, 'char') and key.char
                    else (f"VK_{vk}" if vk else "?"))
            if vk:
                captured['vk'] = vk
                captured['char'] = char
                QTimer.singleShot(0, d.accept)
                return False

        self.rebinding_key = True
        try:
            listener = Listener(on_press=on_press)
            listener.start()
            d.exec()
            listener.stop()
        finally:
            self.rebinding_key = False

        vk, char = captured['vk'], captured['char']
        if not vk:
            return
        if target == 'main':
            self.vk_fishing, self.txt_fishing = vk, char
            self.btn_start.setText(f"Start Fishing ({char})")
        elif target == 'pos':
            self.vk_pos, self.txt_pos = vk, char
            self.visual_tab.set_pos_key_label(char)
        elif target == 'potion':
            self.vk_potion, self.txt_potion = vk, char
            self.btn_potion_key.setText(f"Key: {char}")
            self._on_config_changed()
        elif target == 'train':
            self.vk_train, self.txt_train = vk, char
            self.audio_tab.set_train_key_label(char)
