#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-cellable-build}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SPEC_FILE="${SPEC_FILE:-cellable.spec}"
APP_NAME="${APP_NAME:-Cellable}"
MODEL_BUNDLE="${CELLABLE_MODEL_BUNDLE:-efficientsam_accuracy}"
EXCLUDE_CELLPOSE="${CELLABLE_EXCLUDE_CELLPOSE:-1}"
STRIP_BINARIES="${CELLABLE_STRIP:-0}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

need_cmd conda
need_cmd hdiutil

env_exists() {
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"
}

if ! env_exists; then
  echo "Creating conda env '$ENV_NAME' (python=$PYTHON_VERSION)..."
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

echo "Installing build deps into '$ENV_NAME'..."
echo "Build profile:"
echo "  CELLABLE_MODEL_BUNDLE=$MODEL_BUNDLE (full|balanced|lite|efficientsam_accuracy)"
echo "  CELLABLE_EXCLUDE_CELLPOSE=$EXCLUDE_CELLPOSE (0|1)"
echo "  CELLABLE_STRIP=$STRIP_BINARIES (0|1)"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt
conda run -n "$ENV_NAME" python -m pip install pyinstaller
conda run -n "$ENV_NAME" python -m pip install -e .

echo "Downloading AI models (bundled into the app)..."
conda run -n "$ENV_NAME" python download_models.py

echo "Building macOS .app via PyInstaller ($SPEC_FILE)..."
python3 - <<'PY'
import os
import shutil

for path in ("dist", "build"):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
PY
conda run -n "$ENV_NAME" pyinstaller --noconfirm --clean "$SPEC_FILE"

APP_PATH="dist/${APP_NAME}.app"
if [[ ! -d "$APP_PATH" ]]; then
  echo "Build succeeded but app not found at: $APP_PATH" >&2
  echo "Check dist/ for the actual app name." >&2
  exit 1
fi

if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
  need_cmd codesign
  echo "Codesigning app with identity: $CODESIGN_IDENTITY"
  codesign --deep --force --options runtime --sign "$CODESIGN_IDENTITY" "$APP_PATH"
elif [[ "${ADHOC_SIGN:-1}" == "1" ]]; then
  if command -v codesign >/dev/null 2>&1; then
    echo "Ad-hoc signing app (set ADHOC_SIGN=0 to skip)..."
    codesign --deep --force --sign - "$APP_PATH" || true
  fi
fi

echo "Creating DMG installer..."
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

cp -R "$APP_PATH" "$STAGE_DIR/"
ln -s /Applications "$STAGE_DIR/Applications"

DMG_PATH="dist/${APP_NAME}-macOS.dmg"
rm -f "$DMG_PATH"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE_DIR" -ov -format UDZO "$DMG_PATH"

echo "Done:"
echo "  App: $APP_PATH"
echo "  DMG: $DMG_PATH"
