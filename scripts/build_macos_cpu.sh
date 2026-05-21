#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT_NAME="baseball-motion-app-cpu"

has_conda_environment() {
  conda env list | awk '{ print $1 }' | grep -Fxq "$ENVIRONMENT_NAME"
}

cd "$REPO_ROOT"

if ! has_conda_environment; then
  conda env create -n "$ENVIRONMENT_NAME" -f environment-cpu.yml
fi

if [[ ! -f "pipelines/models/normal/rtmpose_end2end.onnx" ]]; then
  echo "Pose model files are missing. Place the models folder at pipelines/models before building the app." >&2
  exit 1
fi

conda run -n "$ENVIRONMENT_NAME" python -c "import openvino; print('OpenVINO:', openvino.__version__)"
conda run -n "$ENVIRONMENT_NAME" python -c "from pipelines.poseEstimation import setup_backend_device; resolved = setup_backend_device('auto', 'auto'); assert resolved == ('openvino', 'cpu'), resolved; print('Pose backend/device:', resolved)"

npm install
npm run build:ts

conda run -n "$ENVIRONMENT_NAME" python packaging/create_macos_app_icon.py
conda run -n "$ENVIRONMENT_NAME" python -m pip install "pyinstaller>=6.0"
conda run -n "$ENVIRONMENT_NAME" python -m PyInstaller --noconfirm --clean packaging/BaseballMotion.spec

echo "macOS app build complete: $REPO_ROOT/dist/BaseballMotion.app"
