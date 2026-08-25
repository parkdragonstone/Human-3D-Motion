#!/usr/bin/env bash
# Human3DMotion - macOS environment setup (no build)
# Usage: bash scripts/env_setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="human-3d-motion"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda was not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "[INFO] Preparing Conda environment \"$ENV_NAME\" (cpu)"
if conda env list | awk '{ print $1 }' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Conda environment \"$ENV_NAME\" already exists. Reusing it."
else
  conda env create -n "$ENV_NAME" -f environment-cpu.yml
fi

conda run -n "$ENV_NAME" python -c "import openvino; print('OpenVINO:', openvino.__version__)"
conda run -n "$ENV_NAME" python -c "from pipelines.poseEstimation import setup_backend_device; resolved = setup_backend_device('auto', 'auto'); assert resolved in (('onnxruntime', 'mps'), ('openvino', 'cpu')), resolved; print('Pose backend/device:', resolved)"

echo "[INFO] Installing the h3dm command"
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)" | tail -n 1 | tr -d '\r')"
if [[ -z "$ENV_PREFIX" ]]; then
  echo "[ERROR] Could not resolve the Conda environment prefix for \"$ENV_NAME\"." >&2
  exit 1
fi

mkdir -p "$ENV_PREFIX/bin"
COMMAND_PATH="$ENV_PREFIX/bin/h3dm"
cat > "$COMMAND_PATH" <<LAUNCHER
#!/usr/bin/env bash
set -euo pipefail
cd "\${H3DM_REPO_ROOT:-$REPO_ROOT}"
exec "$ENV_PREFIX/bin/python" -u -m webapp.main "\$@"
LAUNCHER
chmod +x "$COMMAND_PATH"
echo "[INFO] Installed h3dm command: $COMMAND_PATH"

echo
echo "[DONE] Environment \"$ENV_NAME\" (cpu) is ready."
echo "       Activate it with: conda activate $ENV_NAME"
echo "       Start the app with: h3dm"
