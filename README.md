# Baseball Motion

Baseball Motion is a local web app for capture, calibration, pose estimation, 3D reconstruction, and kinematics analysis.

## Requirements

- Windows with Conda installed
- Node.js and npm
- Python 3.12, installed through the Conda environment files below
- Optional NVIDIA GPU support for pose estimation

OpenSim Python API is installed with pip as `opensim>=4.6`. A separate OpenSim GUI installer is not required for this app's kinematics pipeline.

## Install

Choose one environment.

GPU:

```bash
conda env create -f environment-gpu.yml
conda activate baseball-motion
```

CPU:

```bash
conda env create -f environment-cpu.yml
conda activate baseball-motion
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

## Run

Start the web app:

```bash
python -m webapp.main
```

Default URL:

```text
https://127.0.0.1:5000
```

The app uses a self-signed development HTTPS certificate by default. If the browser shows a certificate warning, continue to the site for local use.

To run without HTTPS:

```bash
set BASEBALL_MOTION_HTTPS=0
python -m webapp.main
```

Then open:

```text
http://127.0.0.1:5000
```

To use another port:

```bash
set BASEBALL_MOTION_PORT=5001
python -m webapp.main
```

## Docker

CPU:

```bash
docker compose -f docker/docker-compose.yml --profile cpu up --build
```

GPU:

```bash
docker compose -f docker/docker-compose.yml --profile gpu up --build
```

Open:

```text
http://127.0.0.1:5000
```

The GPU container requires Docker with NVIDIA Container Toolkit. Both Docker profiles mount local `recordings`, `webapp_data`, and `pipelines/models` directories into the container. Docker-specific files are under `docker/`.

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

## Useful Commands

Rebuild frontend after TypeScript changes:

```bash
npm run build:ts
```

Check Python syntax for the web app:

```bash
python -m py_compile webapp\presentation\flask_app.py
```

Install only Python packages into an existing environment:

```bash
pip install -r requirements.txt
```
