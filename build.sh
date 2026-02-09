#!/bin/bash

# Configuration
APP_NAME="AutoFisher"
MAIN_SCRIPT="main.py"
ICON_FILE="icon.ico"
SPLASH_AUDIO="Splash_1.wav"

echo "----------------------------------------------------------------"
echo " AutoFisher Builder"
echo "----------------------------------------------------------------"

# 1. Check if required files exist
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: $MAIN_SCRIPT not found!"
    exit 1
fi

if [ ! -f "$ICON_FILE" ]; then
    echo "Warning: $ICON_FILE not found. Building with default icon."
    ICON_ARG=""
else
    echo "Icon found: $ICON_FILE"
    ICON_ARG="--icon=$ICON_FILE"
fi

if [ ! -f "$SPLASH_AUDIO" ]; then
    echo "Error: $SPLASH_AUDIO not found! Cannot bundle default pattern."
    exit 1
else
    echo "Audio Pattern found: $SPLASH_AUDIO"
fi

# 2. Clean previous build artifacts (optional, keeps folder clean)
echo "Cleaning up old build folders..."
rm -rf build dist $APP_NAME.spec

# 3. Run PyInstaller
# --noconsole: Hides the black command window
# --onefile: Creates a single .exe file
# --add-data: Bundles the wav file inside the exe.
#             NOTE: The separator is ';' for Windows. If you are on Linux, change ';' to ':'
echo "Building $APP_NAME.exe ..."

pyinstaller --noconsole --onefile $ICON_ARG \
    --add-data "$SPLASH_AUDIO;." \
    --name "$APP_NAME" \
    "$MAIN_SCRIPT"

# 4. Check result
if [ -f "dist/$APP_NAME.exe" ]; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "SUCCESS! Build complete."
    echo "Your file is located at: dist/$APP_NAME.exe"
    echo "----------------------------------------------------------------"
else
    echo ""
    echo "Build failed. Please check the errors above."
    exit 1
fi