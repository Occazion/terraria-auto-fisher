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

PyInstaller follows the import graph automatically — adding a new module under the project root does not require touching the build script as long as it's imported (directly or transitively) from [main.py](main.py).

## File map

The code is split into flat top-level modules (no package, no `__init__.py`):

| File | Purpose |
|---|---|
| [main.py](main.py) | Entry point only. Configures logging, calls `cleanup_temp_patterns()`, spins up `QApplication`, shows `AppUi`. Owns `__version__`. |
| [paths.py](paths.py) | Filesystem helpers: `resource_path()` (PyInstaller-aware), `get_app_data_dir()` (returns `%APPDATA%/AutoFisher/Patterns`), `cleanup_temp_patterns()` (clears `Splash_*.wav` on startup). |
| [input_control.py](input_control.py) | All user-input I/O. `MouseController` (pyautogui clicks + a robust VK keypress via pynput `Controller`), `tick_potion()` helper used by both workers, `KeyMonitor` (`QObject` wrapper around a pynput `Listener` — emits `keyPressed(vk)` to the GUI thread). |
| [logic.py](logic.py) | Visual-mode brains. `FisherLogic` (4-state machine `INIT → CAST → WAIT → REEL` with configurable re-cast timeout) and `MovementTracker` (3-frame absdiff motion mask). Depends on `input_control.MouseController`. |
| [visual_worker.py](visual_worker.py) | `VisualWorker(QThread)` — grabs a 100×100 region around the bobber with `PIL.ImageGrab`, runs OpenCV motion detection, drives `FisherLogic`, emits preview frames + state to the UI. |
| [audio_workers.py](audio_workers.py) | Three sounddevice `QThread`s: `AudioMonitorWorker` (live volume meter), `AudioTrainerWorker` (mouse-click-bracketed splash recorder), `AudioPatternWorker` (FFT cross-correlation + catch sequence). All magic numbers (sample rates, filter cutoff, RMS window, trigger timing) are named module constants at the top. |
| [ui_common.py](ui_common.py) | Shared UI helpers: button/progress style strings (`STYLE_GREEN`, `STYLE_RED`, `STYLE_ORANGE`, …), default hotkey VK codes (`DEFAULT_VK_FISHING`, …), `safe_int`/`safe_float` (tolerant profile-value coercion that logs and falls back), `vk_to_char()`. |
| [tab_visual.py](tab_visual.py) | `VisualTab(QWidget)` — owns all visual-mode widgets. Public API: `to_runtime_config()`, `to_profile_dict()`, `from_profile_dict(d)`, `update_frame(raw, proc)`, `update_stats(sense, state)`, `set_position(x, y)`, `has_position()`, `set_pos_key_label(char)`. Signals: `config_changed`, `rebind_pos_requested`. |
| [tab_audio.py](tab_audio.py) | `AudioTab(QWidget)` — owns the device combo, gain slider, training section, volume/score meters, threshold slider. Public API: `populate_devices()` (returns count), `get_selected_device()`, `get_gain()`, `set_train_listening()`/`set_train_idle()`, `update_pattern_count(n)`, plus `update_volume`/`update_score`/`set_state_text` for worker hookup. Signals: `config_changed`, `gain_changed`, `device_changed`, `refresh_devices_requested`, `toggle_train_requested`, `rebind_train_requested`, `reset_patterns_requested`. |
| [ui.py](ui.py) | `AppUi(QMainWindow)` — the shell. Owns the profile combo, hotkey-mute button, the two tab widgets, footer (potion controls + Start button), `KeyMonitor`. Routes worker outputs to tabs, profile load/save through `tab.from_profile_dict`/`to_profile_dict`. |

## Architecture

**Threading model** — Qt main thread owns `AppUi` and the two tab widgets. Each background job is a `QThread`:

| Worker | Role |
|---|---|
| `VisualWorker` ([visual_worker.py](visual_worker.py)) | Screen-grab → motion mask → `FisherLogic` → click. |
| `AudioMonitorWorker` ([audio_workers.py](audio_workers.py)) | Live volume meter for the UI. |
| `AudioTrainerWorker` ([audio_workers.py](audio_workers.py)) | Listens to mouse clicks; on every even-numbered click, trims the last 1.5 s of audio around the loudest peak and saves `Splash_N.wav` to `%APPDATA%/AutoFisher/Patterns/`. |
| `AudioPatternWorker` ([audio_workers.py](audio_workers.py)) | Loads pattern WAVs, downsamples to 8 kHz, high-passes at 1 kHz. The PortAudio callback computes FFT cross-correlation and only *requests* a trigger via `_trigger_pending` + `_cooldown_until`. The actual click sequence (`_do_catch_sequence`) runs on the worker's own `run()` thread so audio callbacks never block. |

**Tab encapsulation** — `AppUi` never reaches inside a tab. It calls `visual_tab.to_runtime_config()` / `audio_tab.populate_devices()` / `visual_tab.has_position()` / `audio_tab.set_train_listening()`. Tabs surface user actions through Qt signals (`rebind_pos_requested`, `toggle_train_requested`, `refresh_devices_requested`, etc.) which `AppUi._wire_tab_signals` connects to its handler methods.

**State machine** — `FisherLogic` in [logic.py](logic.py). `WAIT` state re-casts after `cfg['visual_timeout']` seconds (user-configurable, default 10 s). The state-transition delays (1.5 s settle after CAST, 1.0 s WAIT debounce, 2.0 s after REEL) are algorithm constants in `logic._CAST_SETTLE_SEC` etc.

**Config persistence** — profiles in `fishing_profiles.ini` (INI format via `configparser`) at the working directory. `AppUi.loading_profile` blocks `config_changed` propagation while widgets are being populated from disk. Each tab also calls `blockSignals(True)` in `from_profile_dict` for belt-and-braces protection. Profile keys (`vis_th`, `vis_sens`, `audio_th`, etc.) are owned by the tabs' `to_profile_dict`/`from_profile_dict` methods so the on-disk format stays stable when worker config keys change.

**Runtime vs profile keys** — Workers consume `to_runtime_config()` keys (`threshold`, `sensitivity`, `audio_threshold`, …); the INI uses `to_profile_dict()` keys (`vis_th`, `vis_sens`, `audio_th`, …). Each tab owns the mapping in one place.

**Hotkeys** — `KeyMonitor` ([input_control.py](input_control.py)) emits `keyPressed(vk)` from a pynput thread; Qt's queued connection delivers it to `AppUi._handle_hotkey_vk` on the GUI thread. Hotkeys are suppressed when `rebinding_key` is set (while the `_wait_key` modal is open) or when the relevant button is disabled (e.g. Start during training).

**Rebind dialog** — `AppUi._wait_key` does *no* widget work inside the pynput listener thread; it captures the key into a local dict, asks the GUI thread to close the modal via `QTimer.singleShot(0, d.accept)`, then applies all changes after `d.exec()` returns. Touching Qt widgets from the pynput thread causes hangs.

**Resource path** — `paths.resource_path()` resolves bundled assets from `sys._MEIPASS` when running as a PyInstaller `.exe`, and from `os.path.dirname(__file__)` of [paths.py](paths.py) when running from source (which is the project root, where `Splash_1.wav` lives).

**Audio trigger cooldown** — when `AudioPatternWorker` detects a match it sets `_cooldown_until = time.time() + _trigger_action_sec + _TRIGGER_COOLDOWN_PAD_SEC`. The callback ignores matches during this window; the run loop performs the click+wait+click on its own thread. See `_TRIGGER_REEL_WAIT_SEC` / `_TRIGGER_RECAST_WAIT_SEC` constants at the top of [audio_workers.py](audio_workers.py).
