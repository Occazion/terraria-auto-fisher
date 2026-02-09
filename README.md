
---

# 🎣 Terraria AutoFisher (Visual & Audio)

**Version:** 1.8 (Stable + Smart Training)
**Author:** Inspired by alexesmet/terraria-auto-fisher

A high-performance, external fishing bot designed for **Terraria** (and compatible with Minecraft, WoW, etc.). It features two distinct detection modes: **Visual Motion Detection** and **Advanced Audio Pattern Matching**, allowing you to fish in any weather condition or lighting.

---

## ✨ Key Features

* **Dual Modes:**
* 📷 **Visual Mode:** Detects bobber movement using OpenCV motion tracking.
* 🔊 **Audio Mode:** Uses FFT Correlation to match specific sound patterns (Splashes) regardless of volume or background noise.


* **Smart Training System:** "Teach" the bot what your specific fishing splash sounds like with a single click. Supports multiple variations (Fast/Slow splashes).
* **Background Noise Filtering:** Automatically filters out wind, rain, and music using High-Pass filters and Spectral Gating.
* **Profile Manager:** Save and load configurations for different biomes (e.g., "Ocean", "Cavern", "Lava").
* **Safety Features:**
* Randomized reaction times (human-like).
* Auto-timeout (re-casts if no fish is caught in 10s).
* Safe Shutdown (prevents driver crashes).


* **Potion Auto-Drink:** Automatically buffs fishing potions on a timer.

---

## ⚙️ Installation & Prerequisites

### 1. Audio Drivers (Crucial for Audio Mode)

To let the bot "hear" the game sounds without using your microphone, you must install a Loopback driver.

1. Download and Install **[VB-CABLE Virtual Audio Device](https://vb-audio.com/Cable/)** (Free).
2. **Windows Settings:** Set your Playback device to **CABLE Input**.
3. **To hear the game yourself:** Go to Sound Settings -> Recording -> CABLE Output -> Properties -> Listen -> Check **"Listen to this device"** and select your speakers/headphones.

### 2. Running from Source

Ensure you have Python 3.8+ installed.

```bash
# 1. Clone or download this repository
git clone https://github.com/yourusername/AutoFisher.git
cd AutoFisher

# 2. Install dependencies
pip install PyQt6 opencv-python numpy pyautogui pynput sounddevice soundfile scipy pillow

# 3. Run the application
python main.py

```

### 3. Building an Executable (.exe)

If you want to create a standalone file:

```bash
# Windows
./build.sh
# or manually:
pyinstaller --noconsole --onefile --add-data "Splash_1.wav;." --name "AutoFisher" main.py

```

---

## 📖 Usage Guide

### Mode A: 📷 Visual Fishing (Classic)

Best for calm water with no rain/background movement.

1. Launch the app and select the **Visual Mode** tab.
2. Cast your fishing rod in-game.
3. Press the **Set Pos Key** (Default: `V`) button in the app, or press `V` on your keyboard while hovering your mouse over the bobber.
4. Adjust **Threshold** (Movement sensitivity):
* Lower (5-10) = Detects tiny movements.
* Higher (20+) = Ignores rain/wind.


5. Press **Start Fishing** (Default: `F`).

### Mode B: 🔊 Audio Pattern Mode (Recommended)

Best for all conditions. Requires 1 minute of "Training".

**Phase 1: Setup**

1. Go to the **Audio Pattern Mode** tab.
2. Select **CABLE Output (VB-Audio)** in the device dropdown.
3. Look at the "1. Raw Volume" bar. Play a sound in-game. If the bar moves, you are ready.
4. *Tip: Use the Digital Gain slider if the game volume is too low.*

**Phase 2: Training (The "T" Key)**

1. Press **TRAIN MODE** (Default: `T`). The button turns Orange.
2. Go to Terraria. **Manually fish**.
* **Click 1 (Cast):** The bot detects this but ignores it (logs "Casted... WAIT").
* **Click 2 (Catch):** When you hear the *SPLASH* and click to reel in, the bot captures that sound, trims it, and saves it as a pattern.


3. Catch 3-5 fish this way to record different splash variations.
4. Press `T` again to Stop Training.

**Phase 3: Auto Fishing**

1. Set the **Match Threshold** to ~60%.
2. Press **Start Fishing** (Default: `F`).
3. The bot will now compare live audio against your recorded patterns. If the "Max Match Score" bar hits the threshold, it triggers.

---

## 🎮 Hotkeys

| Key | Function | Description |
| --- | --- | --- |
| **F** | Start/Stop Fishing | Toggles the active worker (Visual or Audio). |
| **V** | Set Position | Updates the X/Y coordinates to current mouse position (Visual Mode). |
| **T** | Toggle Training | Starts/Stops the audio recording mode. |

*Note: You can rebind these keys in the GUI.*

---

## 📂 Configuration

* **Profiles:** Your settings (Thresholds, X/Y coords, Gain) are saved in `fishing_profiles.ini`.
* **Audio Patterns:** Recorded splashes are stored in `%APPDATA%/AutoFisher/Patterns/`. These are temporary and cleared when you restart the app (except for the built-in `Splash_1.wav`).

---

## 🔧 Troubleshooting

**Q: The Audio Bar isn't moving!**
A: You selected the wrong device in the dropdown. Try selecting "Stereo Mix" or "CABLE Output". Ensure Windows volume for that device is at 100%.

**Q: Visual mode triggers constantly without fish.**
A: Increase the **Threshold** value. Rain or background animations are triggering the motion detector.

**Q: Audio match score is low (<20%) even on splashes.**
A:

1. Increase **Digital Gain**.
2. Retrain the bot (Press T, catch a few fish).
3. Ensure "Background Music" in Terraria is lowered/off, and "Sound Effects" are maxed.

**Q: Error 0xC0000005 (Access Violation) on close.**
A: Update to v1.8 (Current). This version includes a safe shutdown sequence for audio threads.

---

## ⚖️ Disclaimer

*Use this software at your own risk.* While this bot uses external detection methods (Screen/Audio capture) and does not inject code into the game memory, automating gameplay may violate the Terms of Service of some online games. The author assumes no responsibility for banned accounts.
