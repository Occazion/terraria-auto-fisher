"""Visual Mode tab widget.

Owns all visual-mode UI (position spinners, threshold/sensitivity controls,
preview frames, state label). Exposes a small API so AppUi never has to
touch internal widgets directly.

Signals:
    config_changed         — any widget value changed
    rebind_pos_requested   — user clicked "Set Pos Key"
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QFormLayout, QFrame, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                             QWidget)

from ui_common import safe_int


class VisualTab(QWidget):
    config_changed = pyqtSignal()
    rebind_pos_requested = pyqtSignal()

    def __init__(self, pos_key_label: str = 'V', parent=None):
        super().__init__(parent)
        self._build_ui(pos_key_label)
        self._wire_signals()

    # ----------------- construction -----------------

    def _build_ui(self, pos_key_label: str) -> None:
        layout = QVBoxLayout(self)

        # Preview row
        prev_row = QHBoxLayout()
        self.lbl_raw = self._make_preview("Raw")
        self.lbl_proc = self._make_preview("Mask")
        prev_row.addWidget(self.lbl_raw)
        prev_row.addWidget(self.lbl_proc)
        layout.addLayout(prev_row)

        # Live stats
        self.prog_bar = QProgressBar()
        self.lbl_state = QLabel("State: IDLE")
        layout.addWidget(self.prog_bar)
        layout.addWidget(self.lbl_state)

        # Form controls
        form = QFormLayout()
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 5000)
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 5000)
        self.btn_pos_key = QPushButton(f"Set Pos Key ({pos_key_label})")
        self.btn_pos_key.clicked.connect(self.rebind_pos_requested.emit)

        self.spin_th = QSpinBox()
        self.spin_th.setRange(0, 255)
        self.spin_th.setValue(10)

        self.spin_sens = QSpinBox()
        self.spin_sens.setRange(0, 2000)
        self.spin_sens.setValue(50)

        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(2, 120)
        self.spin_timeout.setValue(10)
        self.spin_timeout.setSuffix("s")

        form.addRow("X / Y:", self._h_box(self.spin_x, self.spin_y))
        form.addRow("Pos Hotkey:", self.btn_pos_key)
        form.addRow("Threshold:", self.spin_th)
        form.addRow("Sensitivity:", self.spin_sens)
        form.addRow("Re-cast Timeout:", self.spin_timeout)
        layout.addLayout(form)

    def _wire_signals(self) -> None:
        for w in (self.spin_x, self.spin_y, self.spin_th, self.spin_sens, self.spin_timeout):
            w.valueChanged.connect(self.config_changed.emit)

    @staticmethod
    def _make_preview(tooltip: str) -> QLabel:
        lbl = QLabel()
        lbl.setFixedSize(100, 100)
        lbl.setFrameShape(QFrame.Shape.Box)
        lbl.setToolTip(tooltip)
        return lbl

    @staticmethod
    def _h_box(w1: QWidget, w2: QWidget) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(w1)
        layout.addWidget(w2)
        return w

    # ----------------- public API -----------------

    def set_pos_key_label(self, char: str) -> None:
        self.btn_pos_key.setText(f"Set Pos Key ({char})")

    def set_position(self, x: int, y: int) -> None:
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)

    def has_position(self) -> bool:
        return self.spin_x.value() != 0 or self.spin_y.value() != 0

    def to_runtime_config(self) -> dict:
        """Worker-facing config keys."""
        return {
            'x': self.spin_x.value(),
            'y': self.spin_y.value(),
            'threshold': self.spin_th.value(),
            'sensitivity': self.spin_sens.value(),
            'visual_timeout': self.spin_timeout.value(),
        }

    def to_profile_dict(self) -> dict:
        """INI-file keys (stable for backward compat)."""
        return {
            'x': str(self.spin_x.value()),
            'y': str(self.spin_y.value()),
            'vis_th': str(self.spin_th.value()),
            'vis_sens': str(self.spin_sens.value()),
            'vis_timeout': str(self.spin_timeout.value()),
        }

    def from_profile_dict(self, d) -> None:
        """Load values from an INI section (or any dict-like). Caller is
        expected to suppress config_changed via its own flag if needed."""
        self.blockSignals(True)
        try:
            self.spin_x.setValue(safe_int(d, 'x', 0))
            self.spin_y.setValue(safe_int(d, 'y', 0))
            self.spin_th.setValue(safe_int(d, 'vis_th', 10))
            self.spin_sens.setValue(safe_int(d, 'vis_sens', 50))
            self.spin_timeout.setValue(safe_int(d, 'vis_timeout', 10))
        finally:
            self.blockSignals(False)

    # ----------------- worker callbacks -----------------

    def update_frame(self, raw: QImage, proc: QImage) -> None:
        self.lbl_raw.setPixmap(QPixmap.fromImage(raw))
        self.lbl_proc.setPixmap(QPixmap.fromImage(proc))

    def update_stats(self, sense: float, state: str) -> None:
        self.lbl_state.setText(f"State: {state}")
        self.prog_bar.setValue(min(100, int(sense * 100)))

    def set_state_text(self, text: str) -> None:
        self.lbl_state.setText(text)
