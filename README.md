# Human3DMotion (H3DM)

<p align="center">
  <img src="images/human-3d-motion.png" alt="Human 3D Motion" width="480">
</p>

H3DM is a local web app for capture, calibration, pose estimation, 3D reconstruction, and kinematics analysis.

## Requirements

- Windows with Conda installed
- Node.js and npm
- Python 3.12, installed through the Conda environment files below
- Optional NVIDIA GPU support for pose estimation

## Install

Choose one environment.

GPU on Windows PowerShell:

```powershell
.\scripts\setup_gpu_env.ps1
conda activate human-3d-motion
```

GPU on Windows CMD:

```bat
scripts\setup_gpu_env.cmd
conda activate human-3d-motion
```

CPU:

```bash
conda env create -f environment-cpu.yml
conda activate human-3d-motion
scripts\install_h3dm_command.cmd
```

Install frontend dependencies and build TypeScript:

```bash
npm install
npm run build:ts
```

Download the model files from Google Drive:

```text
https://drive.google.com/drive/folders/1aJ6LuDQF4ahWF9E_gj_r4981sVExRAN8?usp=sharing
```

Place the downloaded `models` folder here:

```text
pipelines/models
```

The pose pipeline expects model files under `pipelines/models`, for example `pipelines/models/normal/rtmpose_end2end.onnx`.

`node_modules/`, `webapp_data/`, and generated recording/analysis outputs are local artifacts and should not be committed.

## Windows Executable

To create a GPU Windows executable from CMD, place the model files at
`pipelines/models` first and run one command from the repository root:

```bat
build_exe_gpu.cmd
```

The build command creates or reuses the dedicated `human-3d-motion` Conda
environment, normalizes ONNX Runtime to `onnxruntime-gpu`, verifies CUDA pose
backend selection, installs frontend/build dependencies, builds TypeScript, and
generates the executable icon from `images/human-3d-motion.png`.

To create a CPU Windows executable instead, run:

```bat
build_exe_cpu.cmd
```

The CPU build command creates or reuses the dedicated
`human-3d-motion` Conda environment and verifies OpenVINO CPU pose
backend selection. Both build commands create:

```text
dist\Human3DMotion\Human3DMotion.exe
```

The output is a PyInstaller one-folder build. Keep the files beside the `.exe`
in `dist\Human3DMotion` together when moving the app.

To create a macOS CPU app, place the model files at `pipelines/models` first
and run:

```bash
bash build_macos_cpu.sh
```

The macOS build command creates or reuses the dedicated
`human-3d-motion-app-cpu` Conda environment, verifies OpenVINO CPU pose backend
selection, generates an `.icns` icon from `images/human-3d-motion.png`, and
creates:

```text
dist/Human3DMotion.app
```

The macOS app is built locally for the architecture of the Python environment
that runs the build. Code signing and notarization are not part of this build
command.

## Run

Start the web app from the repository root:

```bat
h3dm
```

Default URL format:

```text
https://<internal-ip>:9090
```

The app uses a self-signed development HTTPS certificate by default. If the
browser shows a certificate warning, continue to the site for local use. The
`h3dm` command opens the internal IP URL in your default browser. Set
`HUMAN_3D_MOTION_BROWSER_HOST` or `HUMAN_3D_MOTION_PUBLIC_URL` to override the
opened address.

To run without HTTPS:

```bat
set HUMAN_3D_MOTION_HTTPS=0
h3dm
```

Then open:

```text
http://<internal-ip>:9090
```

To use another port:

```bat
set HUMAN_3D_MOTION_PORT=5001
h3dm
```

To keep the browser closed on launch:

```bat
set HUMAN_3D_MOTION_OPEN_BROWSER=0
h3dm
```

## Demo Data

Demo files are available from Google Drive:

```text
https://drive.google.com/drive/folders/1JD7Ye4nBwJI8rVy0jvVXBtfj6-yEsI_9?usp=drive_link
```

Download the demo folder to a local path, then open the Analysis page and select that folder as the Analysis session root. The demo data should follow the same session filename format described below.

## First Setup

1. Open the Capture or Calibration page and select the storage root.
2. Configure camera mode and camera count.
3. Capture or place session videos using the expected session filename format:

```text
name_height_weight_hand_YYYYMMDD_HHMMSS_cam01.mp4
name_height_weight_hand_YYYYMMDD_HHMMSS_cam02.mp4
```

`.mp4` and `.avi` session videos are supported.

4. Open the Analysis page and select the Analysis session root.
5. Select a session and run analysis.

Analysis keeps its own root path separate from Capture/Calibration storage root.