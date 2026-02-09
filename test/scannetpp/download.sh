#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
URL="https://huggingface.co/datasets/nyu-visionx/VSI-Bench/resolve/main/scannetpp.zip?download=true"
ZIP_FILE="$SCRIPT_DIR/scannetpp.zip"

echo "Downloading scannetpp.zip..."
curl -L -o "$ZIP_FILE" "$URL"

echo "Extracting mp4 files..."
unzip -j -o "$ZIP_FILE" "*.mp4" -d "$SCRIPT_DIR"

rm "$ZIP_FILE"
echo "Done. Files extracted to $SCRIPT_DIR/"
