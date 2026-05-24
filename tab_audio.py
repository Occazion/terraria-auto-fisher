"""Audio Pattern Mode tab widget.

Owns the device combo, gain slider, training section, volume/score meters,
and the match-threshold slider.

Signals:
    config_changed             — any value that goes into runtime cfg changed
    gain_changed               — gain slider moved (AppUi needs to update monitor)
    device_changed             — combo selection changed (AppUi restarts monitor)
    refresh_devices_requested  — user clicked ↻
    toggle_train_requested     — user clicked TRAIN button
    rebind_train_requested     — user clicked "Set Key" for TRAIN hotkey
    reset_patterns_requested   — user clicked "Reset Patterns"
"""

import logging

import sounddevice as sd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QProgressBar, QPushButton, QSlider,
                             QVBoxLayout, QWidget)

from ui_common import (STYLE_BAR_BLUE, STYLE_BAR_GREEN, STYLE_LISTENING,
                       STYLE_ORANGE, safe_float, safe_int)


class AudioTab(QWidget):
    config_changed = pyqtSignal()
    gain_changed = pyqtSignal()
    device_changed = pyqtSignal()
    refresh_devices_requested = pyqtSignal()
    toggle_train_requested = pyqtSignal()
    rebind_train_requested = pyqtSignal()
    reset_patterns_requested = pyqtSignal()

    def __init__(self, train_key_label: str = 'T', parent=None):
        super().__init__(parent)
        self._train_key_label = train_key_label
        self._build_ui()
        self._wire_signals()

    # ----------------- construction -----------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select Loopback Device:"))
        dev_row = QHBoxLayout()
        self.combo_devices = QComboBox()
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setToolTip("Refresh device list")
        self.btn_refresh.setFixedWidth(30)
        dev_row.addWidget(self.combo_devices, 1)
        dev_row.addWidget(self.btn_refresh)
        layout.addLayout(dev_row)

        # Gain slider
        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Digital Gain:"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(10, 100)
        self.slider_gain.setValue(10)
        self.lbl_gain_val = QLabel("1.0x")
        gain_row.addWidget(self.slider_gain)
        gain_row.addWidget(self.lbl_gain_val)
        layout.addLayout(gain_row)

        layout.addWidget(self._build_train_group())

        # Volume / score meters
        layout.addWidget(QLabel("1. Raw Volume:"))
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setFormat("%v %")
        self.volume_bar.setStyleSheet(STYLE_BAR_BLUE)
        layout.addWidget(self.volume_bar)

        layout.addWidget(QLabel("2. Max Match Score:"))
        self.audio_bar = QProgressBar()
        self.audio_bar.setRange(0, 100)
        self.audio_bar.setFormat("%v %")
        self.audio_bar.setStyleSheet(STYLE_BAR_GREEN)
        layout.addWidget(self.audio_bar)

        self.lbl_state = QLabel("Status: Ready")
        layout.addWidget(self.lbl_state)

        # Match threshold slider
        form = QFormLayout()
        self.slider_th = QSlider(Qt.Orientation.Horizontal)
        self.slider_th.setRange(0, 100)
        self.slider_th.setValue(60)
        self.lbl_th_val = QLabel("60%")
        form.addRow("Match Threshold:", self._h_box(self.slider_th, self.lbl_th_val))
        layout.addLayout(form)

    def _build_train_group(self) -> QGroupBox:
        group = QGroupBox("Training")
        layout = QVBoxLayout(group)

        self.lbl_pattern_count = QLabel("Patterns Available: 0")
        layout.addWidget(self.lbl_pattern_count)

        btn_row = QHBoxLayout()
        self.btn_train = QPushButton(f"TRAIN MODE ({self._train_key_label})")
        self.btn_train.setStyleSheet(STYLE_ORANGE)
        self.btn_train_key = QPushButton("Set Key")
        self.btn_train_key.setToolTip("Rebind the TRAIN hotkey")
        self.btn_reset = QPushButton("Reset Patterns")
        for w in (self.btn_train, self.btn_train_key, self.btn_reset):
            btn_row.addWidget(w)
        layout.addLayout(btn_row)

        layout.addWidget(QLabel(
            "1. Enable Train Mode.\n"
            "2. Cast manually (Click 1 - IGNORED).\n"
            "3. Catch fish (Click 2 - SAVED)."
        ))
        return group

    def _wire_signals(self) -> None:
        self.combo_devices.currentIndexChanged.connect(self.device_changed.emit)
        self.btn_refresh.clicked.connect(self.refresh_devices_requested.emit)
        self.btn_train.clicked.connect(self.toggle_train_requested.emit)
        self.btn_train_key.clicked.connect(self.rebind_train_requested.emit)
        self.btn_reset.clicked.connect(self.reset_patterns_requested.emit)
        self.slider_gain.valueChanged.connect(self._on_gain_slider)
        self.slider_th.valueChanged.connect(self._on_threshold_slider)

    def _on_gain_slider(self, v: int) -> None:
        self.lbl_gain_val.setText(f"{v / 10:.1f}x")
        self.config_changed.emit()
        self.gain_changed.emit()

    def _on_threshold_slider(self, v: int) -> None:
        self.lbl_th_val.setText(f"{v}%")
        self.config_changed.emit()

    @staticmethod
    def _h_box(w1: QWidget, w2: QWidget) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(w1)
        layout.addWidget(w2)
        return w

    # ----------------- public API: state / data -----------------

    def get_gain(self) -> float:
        return self.slider_gain.value() / 10.0

    def get_selected_device(self):
        """Returns the userData of the current combo item (the device index, or None)."""
        return self.combo_devices.currentData()

    def populate_devices(self) -> int:
        """Re-enumerate sounddevice input devices.

        Preserves the previous selection if still present. Returns the number of
        devices found. Updates the state label to a warning if none.
        """
        prev_idx = self.combo_devices.currentData()
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
            self.lbl_state.setText(
                "Status: No audio input devices. Install VB-CABLE or check drivers."
            )
        else:
            for i in range(self.combo_devices.count()):
                if self.combo_devices.itemData(i) == prev_idx:
                    self.combo_devices.setCurrentIndex(i)
                    break
        self.combo_devices.blockSignals(False)
        return found

    def to_runtime_config(self) -> dict:
        return {
            'audio_threshold': self.slider_th.value(),
            'audio_gain': self.get_gain(),
        }

    def to_profile_dict(self) -> dict:
        return {
            'audio_th': str(self.slider_th.value()),
            'audio_gain': str(self.get_gain()),
        }

    def from_profile_dict(self, d) -> None:
        self.blockSignals(True)
        try:
            self.slider_th.setValue(safe_int(d, 'audio_th', 60))
            self.slider_gain.setValue(int(safe_float(d, 'audio_gain', 1.0) * 10))
            # blockSignals on the tab doesn't auto-update labels — refresh manually
            self.lbl_th_val.setText(f"{self.slider_th.value()}%")
            self.lbl_gain_val.setText(f"{self.get_gain():.1f}x")
        finally:
            self.blockSignals(False)

    # ----------------- public API: train state -----------------

    def set_train_listening(self) -> None:
        self.btn_train.setText("LISTENING (Cast then Catch)")
        self.btn_train.setStyleSheet(STYLE_LISTENING)

    def set_train_idle(self) -> None:
        self.btn_train.setText(f"TRAIN MODE ({self._train_key_label})")
        self.btn_train.setStyleSheet(STYLE_ORANGE)

    def set_train_key_label(self, char: str) -> None:
        self._train_key_label = char
        self.btn_train.setText(f"TRAIN MODE ({char})")

    def update_pattern_count(self, count: int) -> None:
        self.lbl_pattern_count.setText(f"Patterns Available: {count}")

    # ----------------- public API: worker output -----------------

    def update_volume(self, v: int) -> None:
        self.volume_bar.setValue(v)

    def reset_volume(self) -> None:
        self.volume_bar.setValue(0)

    def update_score(self, v: int) -> None:
        self.audio_bar.setValue(v)

    def set_state_text(self, text: str) -> None:
        self.lbl_state.setText(text)
