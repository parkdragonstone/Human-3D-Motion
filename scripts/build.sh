#!/usr/bin/env bash
# Human3DMotion - macOS app build
# Usage: bash scripts/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="human-3d-motion"

cd "$REPO_ROOT"

if [[ ! -f "pipelines/models/normal/rtmpose_end2end.onnx" ]]; then
  echo "[ERROR] Pose model files are missing." >&2
  echo "        Place the models folder at pipelines/models before building." >&2
  exit 1
fi

bash "$SCRIPT_DIR/env_setup.sh"

echo "[INFO] Building frontend assets"
npm install
npm run build:ts

echo "[INFO] Preparing OpenVINO dylibs and the app icon"
conda run -n "$ENV_NAME" python scripts/fix_openvino_macos_dylibs.py
conda run -n "$ENV_NAME" python packaging/create_macos_app_icon.py

echo "[INFO] Building the app"
conda run -n "$ENV_NAME" python -m pip install "pyinstaller>=6.0"
conda run -n "$ENV_NAME" python -m PyInstaller --noconfirm --clean packaging/Human3DMotion.spec

echo
echo "[DONE] macOS app build complete:"
echo "       $REPO_ROOT/dist/Human3DMotion.app"
