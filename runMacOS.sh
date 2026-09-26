#!/bin/bash
# Run Unleashed locally on a Mac (UI work, building facesets, small previews).
# Heavy video renders belong on Colab: here everything runs on the CPU.
#
#   ./runMacOS.sh            first run: makes .venv and installs the packages,
#                            then starts the app (the models download once, ~2.5 GB)
#   ./runMacOS.sh --update   reinstall the packages (after requirements.txt changed
#                            this happens by itself)
#
# The UI opens in the browser at http://127.0.0.1:7860 (no public share link;
# see server_share in config.yaml). Ctrl+C stops it.

set -e
cd "$(dirname "$0")"

VENV_DIR=".venv"

# Python 3.12 or 3.13 (3.11 works too). Not 3.14: onnxruntime / insightface have
# no builds for it on Intel Macs.
PYTHON=""
for candidate in python3.12 python3.13 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "Python 3.11-3.13 not found. Install one, e.g.:  brew install python@3.12"
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Note: ffmpeg not found (needed for video). Install it with:  brew install ffmpeg"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating $VENV_DIR with $PYTHON ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# (Re)install when requirements.txt changed since the last install, or on --update.
STAMP="$VENV_DIR/.requirements.sha"
WANT="$(shasum requirements.txt | cut -d' ' -f1)"
if [ "$1" == "--update" ] || [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$WANT" ]; then
    echo "Installing packages (insightface is compiled on the first install: a few minutes) ..."
    "$VENV_DIR/bin/python" -m pip install -q --upgrade pip
    "$VENV_DIR/bin/python" -m pip install -r requirements.txt
    echo "$WANT" > "$STAMP"
fi

export NO_ALBUMENTATIONS_UPDATE=1
exec "$VENV_DIR/bin/python" run.py
