# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the application
python main.py

# Install dependencies
pip install -r requirements.txt

# Build standalone .exe (Windows, requires Git Bash or WSL)
./build.sh
# or manually:
pyinstaller --noconsole --onefile --icon=icon.ico --add-data "Splash_1.wav;." --name "AutoFisher" main.py
```

## Architecture

The entire application lives in a single file: `main.py`. There is no package structure.

**Threading model** — the GUI (`AppUi`, a `QMainWindow`) runs on the Qt main thread and spawns `QThread` workers:

| Worker | Role |
|---|---|
| `VisualWorker` | Captures a screen region around the bobber with `PIL.ImageGrab`, runs OpenCV motion detection (`MovementTracker`), and drives `FisherLogic` to click at the right time |
| `AudioMonitorWorker` | Reads the loopback audio device and emits a live volume level for the UI meter |
| `AudioTrainerWorker` | Records audio into a ring buffer while listening for mouse clicks; on even-numbered clicks it trims the last 1.5 s of audio around the loudest peak and saves a `Splash_N.wav` pattern to `%APPDATA%/AutoFisher/Patterns/` |
| `AudioPatternWorker` | Loads all pattern WAVs, downsamples to 8 kHz, high-pass filters at 1 kHz, and in its callback computes FFT cross-correlation between live audio and each pattern; fires a mouse click when the best score exceeds `audio_threshold` |

**State machine** — `FisherLogic` owns a simple 4-state machine (`INIT → CAST → WAIT → REEL → CAST …`) used by `VisualWorker`. A 10-second timeout in `WAIT` forces a re-cast.

**Config persistence** — profiles are stored in `fishing_profiles.ini` (INI format, `configparser`) in the working directory. The `loading_profile` lock flag on `AppUi` prevents spurious `update_config` calls while a profile is being loaded into the widgets.

**Resource path** — `resource_path()` resolves bundled assets (e.g. `Splash_1.wav`) from `sys._MEIPASS` when running as a PyInstaller `.exe`, falling back to the current directory otherwise.

**Input control** — `MouseController` wraps `pyautogui` for clicks; `KeyMonitor` uses a `pynput` `Listener` thread to emit global hotkey VK codes to the Qt signal/slot system.
