"""Screen-capture motion detection worker (Visual Mode)."""

import time

import cv2
import numpy as np
from PIL import ImageGrab, ImageQt
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from input_control import tick_potion
from logic import FisherLogic, MovementTracker


# Half-width of the screen region grabbed around the bobber coordinates.
# (Full bbox is 100x100 px centered on x/y.)
_GRAB_RADIUS_PX = 50
_FRAME_INTERVAL_SEC = 0.05  # ~20 FPS capture


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
            bbox = (cx - _GRAB_RADIUS_PX, cy - _GRAB_RADIUS_PX,
                    cx + _GRAB_RADIUS_PX, cy + _GRAB_RADIUS_PX)

            try:
                pil_img = ImageGrab.grab(bbox=bbox)
            except OSError:
                time.sleep(0.1)
                continue

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            diff_img = self.tracker.get_diff(gray, int(self.cfg['threshold']))

            count = cv2.countNonZero(diff_img)
            area = (_GRAB_RADIUS_PX * 2) ** 2
            sense_val = count * int(self.cfg['sensitivity']) / area

            state_desc = self.logic.update(sense_val)
            self._handle_potions()

            q_raw = ImageQt.ImageQt(pil_img)
            h, w = diff_img.shape
            q_proc = QImage(diff_img.data, w, h, w, QImage.Format.Format_Grayscale8)

            self.image_processed.emit(q_raw, q_proc)
            self.stats_updated.emit(sense_val, state_desc)
            time.sleep(_FRAME_INTERVAL_SEC)

    def _handle_potions(self) -> None:
        self.potion_timer, signal_val = tick_potion(self.cfg, self.potion_timer)
        self.potion_drank.emit(signal_val)

    def stop(self) -> None:
        self.running = False
        self.wait()
