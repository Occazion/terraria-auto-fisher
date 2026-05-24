"""Visual-mode building blocks: motion tracker + fishing state machine."""

import time

import cv2
import numpy as np

from input_control import MouseController


# State machine transition delays (seconds). These describe the algorithm,
# not the user's preference — leave hardcoded.
_CAST_SETTLE_SEC = 1.5   # after a cast, ignore motion for this long
_WAIT_DEBOUNCE_SEC = 1.0  # ignore a too-fast bobber dip after cast
_REEL_SETTLE_SEC = 2.0   # after reeling, wait this long before next cast


class FisherLogic:
    """4-state machine: INIT → CAST → WAIT → REEL → CAST.

    Driven by a "sense_level" value (motion intensity). A configurable timeout
    in WAIT forces a re-cast so the bot never gets stuck on a stale hook.
    """

    def __init__(self, timeout: float = 10.0):
        self.state = "INIT"
        self.last_action = time.time()
        self.timeout = timeout

    def update(self, sense_level: float) -> str:
        now = time.time()

        # Re-cast if waiting too long with no bite.
        if self.state == "WAIT" and (now - self.last_action) > self.timeout:
            self._switch("REEL")
            MouseController.click()
            return "TIMEOUT! Resetting..."

        if self.state == "INIT":
            if sense_level > 1:
                self._switch("CAST")
                MouseController.click()
        elif self.state == "CAST":
            if (now - self.last_action) > _CAST_SETTLE_SEC and sense_level < 1:
                self._switch("WAIT")
        elif self.state == "WAIT":
            if (now - self.last_action) > _WAIT_DEBOUNCE_SEC and sense_level > 1:
                self._switch("REEL")
                MouseController.click()
        elif self.state == "REEL":
            if (now - self.last_action) > _REEL_SETTLE_SEC and sense_level < 1:
                self._switch("CAST")
                MouseController.click()
        return self.state

    def _switch(self, new_state: str) -> None:
        self.state = new_state
        self.last_action = time.time()


class MovementTracker:
    """3-frame absolute-difference motion mask, OR'd and thresholded.

    Keeps a rolling buffer; once it has at least 3 frames, returns a binary
    mask of pixels that moved between t-2, t-1, and t.
    """

    def __init__(self, size: int = 3):
        self.buffer = []
        self.size = size

    def get_diff(self, img, threshold: int):
        self.buffer.append(img)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
        if len(self.buffer) < 3:
            return np.zeros_like(img)
        t0, t1, t2 = self.buffer[-3:]
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        res = cv2.bitwise_or(d1, d2)
        _, res = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY)
        return res
