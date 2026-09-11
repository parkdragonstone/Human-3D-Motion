# Human3DMotion (H3DM)

<p align="center">
  <img src="images/human-3d-motion.png" alt="Human 3D Motion" width="480">
</p>

H3DM is a local web app for capture, calibration, pose estimation, 3D reconstruction, and kinematics analysis.

## Requirements

- Windows or macOS with Conda installed
- Node.js and npm
- Python 3.12, installed through the Conda environment files below
- Optional NVIDIA GPU support for pose estimation (Windows only)

## Install

Environment setup only, without building the app.

Windows (CMD or PowerShell), from the repository root:

```bat
scripts\env_setup.cmd
```

The script auto-detects the mode: GPU when `nvidia-smi` is available, CPU
otherwise. Force one explicitly with `scripts\env_setup.cmd gpu` or
`scripts\env_setup.cmd cpu`.

macOS (CPU):

```bash
bash scripts/env_setup.sh
```

Both scripts create or reuse the `human-3d-motion` Conda environment from
`environment-gpu.yml` / `environment-cpu.yml`, verify the pose backend, and
install the `h3dm` command into the environment.

```bash
conda activate human-3d-motion
```

Install frontend dependencies and build TypeScript:

```bash
npm install
npm run build:ts
```

### Pose models

Download the model files from Google Drive: [pre-trained](https://drive.google.com/uc?export=download&id=1pTMlT4czh1uv_pwM69p71C3RDlpmPUB4&confirm=t)

Place the downloaded `models` folder here and expects this layout:

```text
pipelines/models
pipelines/models/normal/detector_end2end.onnx   YOLOX-m  (COCO)
pipelines/models/normal/rtmpose_end2end.onnx    RTMPose-m
```

The detector is YOLOX's own ONNX release and can be re-fetched directly; save it as
`detector_end2end.onnx`:

```text
https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx
```

### VideoPose3D lifting weights

Only needed for automatic calibration on the Analysis page. The checkpoint is **not**
bundled with this project: it is published by Meta under CC BY-NC 4.0, which does not
permit commercial use, so you have to download it yourself.


The file is about 65 MB and must end up at exactly this path: [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/main/INFERENCE.md)

```text
pipelines/models/videopose3d/pretrained_h36m_detectron_coco.bin
```

Lifting runs on PyTorch; the CPU build is enough. Without the file, sessions that have no
calibration file stop with `videopose3d_checkpoint_not_found`; sessions that do have a
calibration are unaffected.

`node_modules/`, `webapp_data/`, and generated recording/analysis outputs are local artifacts and should not be committed.


## Build

Place the model files at `pipelines/models` first. Each build script runs the
matching environment setup for you, so running `env_setup` beforehand is
optional.

```bat
scripts\build.cmd
```

Like the setup script, it auto-detects GPU/CPU; pass `gpu` or `cpu` to force a
mode. The build installs frontend dependencies, builds TypeScript, generates the
executable icon from `images/human-3d-motion.png`, and creates:

```text
dist\Human3DMotion\Human3DMotion.exe
```

The output is a PyInstaller one-folder build. Keep the files beside the `.exe`
in `dist\Human3DMotion` together when moving the app.

macOS app (CPU):

```bash
bash scripts/build.sh
```

It generates an `.icns` icon from `images/human-3d-motion.png` and creates:

```text
dist/Human3DMotion.app
```

```text
lsof -ti tcp:9090 | xargs kill
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

To use another port:

```bat
set HUMAN_3D_MOTION_PORT=5001
h3dm
```


## Docker

Runs the app in a container so you do not have to set up Conda. Two image
variants are built from the same `docker/Dockerfile`: `cpu` and `gpu`.

| Platform | What to use |
| --- | --- |
| Linux / Windows with an NVIDIA GPU | Docker, `gpu` profile |
| Linux / Windows without a GPU | Docker, `cpu` profile |
| **macOS** | **Native (`h3dm`), not Docker**|


### Before the first build

Place the pose models at `pipelines/models` first (see
[Pose models](#pose-models)). They are not in the repository, and the image
copies them in at build time; without them analysis fails with
`detector_model_not_found`.

The `gpu` profile additionally needs the NVIDIA driver and the NVIDIA Container
Toolkit on the host. On Windows that means Docker Desktop with the WSL2 backend.
No CUDA toolkit install is required — `onnxruntime-gpu[cuda,cudnn]` pulls the
CUDA runtime in as Python packages.

### Run

```bash
docker compose -f docker/docker-compose.yml --profile cpu up -d --build
docker compose -f docker/docker-compose.yml --profile gpu up -d --build

docker compospe -f docker/docker-compose.yml down
```

```text
https://<internal-ip>:9090
```

### Configuration

Copy the sample and edit it. Compose reads `.env` from the directory holding the
Compose file, so it must live at `docker/.env`:

```bash
cp docker/.env.example docker/.env
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `H3DM_DATA` | Host folder for recordings and analysis output | `../data` |
| `H3DM_PUBLIC_URL` | Pins the address used in phone-capture QR codes | request host |
| `H3DM_PORT` | Published port on the host | `9090` |
| `H3DM_THREADS` | Inference threads | `4` |
| `TZ` | Time zone used in session folder names | `Asia/Seoul` |

### Open the UI on the LAN address, not localhost

The QR codes for phone capture point at whatever address you used to reach the
app. Open `https://localhost:9090` and the QR codes say `localhost`, which no
phone can reach. Either browse to the machine's LAN address:

```text
https://<internal-ip>:9090
```

or pin it once in `docker/.env`:

```bash
H3DM_PUBLIC_URL=https://<internal-ip>:9090
```

Find the address with `ipconfig getifaddr en0` (macOS), `hostname -I` (Linux),
or `ipconfig` (Windows).

### Do not run the native app and the container at the same time

Both bind port 9090 and requests land on whichever won the bind, which is
confusing to debug. Stop one, or set `H3DM_PORT` to move the container.

### Data

Everything the app writes lives on the host under `H3DM_DATA`, so removing the
container does not delete recordings or analysis output:

```text
<H3DM_DATA>/recordings/     capture sessions and analysis results
<H3DM_DATA>/webapp_data/    settings.json
<H3DM_DATA>/tmp/            temporary archives built by Export
```


## Demo Data

Demo files are available from Google Drive: [demo](https://drive.google.com/drive/folders/1JD7Ye4nBwJI8rVy0jvVXBtfj6-yEsI_9?usp=drive_link)


```text
intrinsic calibration info
CharucoBoard | DICT40x40 | 75 | 60 | 4 | 6
Extrinsic Calibration Info
Object
0.0, 0.0, 0.0
0.492, 0.0, 0.0
0.0, 0.0, 0.45
0.492, 0.0, 0.45
0.0, 0.492, 0.0
0.492, 0.492, 0.0
0.0, 0.492,0.45
0.492, 0.492, 0.45
```

Download the demo folder to a local path, then open the Analysis page and select that folder as the Analysis session root. The demo data should follow the same session filename format described below.


## Reference

1. Zeni Jr, J. A., Richards, J. G., & Higginson, J. (2008). Two simple methods for determining gait events during treadmill and overground walking using kinematic data. Gait & posture, 27(4), 710-714. / Detect Gait Events toe off, heel strike Using Heel, Toe, Sacrum (we use Hip instaed)

## Built On

H3DM adapts work from the following open-source projects. Each row lists what this
repository actually takes from the project and where that lands in the code.

| Project | Used for | Where | License |
| --- | --- | --- | --- |
| [Pose2Sim](https://github.com/perfanalytics/pose2sim) | OpenSim models, scaling/IK setup XML and marker sets; the weighted multi-view triangulation approach; person sorting from the companion Sports2D project | `pipelines/OpenSim_Setup/`, `pipelines/kinematics/`, `pipelines/reconstruction/` | BSD-3-Clause |
| [OpenCap](https://github.com/opencap-org/opencap-core) | LSTM marker augmenter (v0.3 lower/upper) that adds anatomical markers to the triangulated TRC | `pipelines/MarkerAugmenter/`, `pipelines/markerAugmentation.py` | Apache-2.0 |
| [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) | Temporal dilated-convolution model that lifts 2D keypoints to 3D, used to obtain per-camera bone directions during automatic calibration | `pipelines/calibration/keypoints/lift3d.py` | **CC BY-NC 4.0 (non-commercial)** |
| [lab-camera-dynamic-calibrator](https://github.com/flodelaplace/lab-camera-dynamic-calibrator) | Markerless extrinsic calibration from human motion: linear solve from bone orientations, bundle adjustment, metric scaling | `pipelines/calibration/keypoints/` | MIT |
| [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) via [rtmlib](https://github.com/Tau-J/rtmlib) | 2D whole-body pose estimation (Halpe-26). `rtmlib` runs the bundled ONNX model without the full MMPose stack | `pipelines/pose_estimation/models.py`, `pipelines/models/normal/rtmpose_end2end.onnx` | Apache-2.0 |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) via [rtmlib](https://github.com/Tau-J/rtmlib) | Person detection; the boxes RTMPose runs on. The bundled weights are YOLOX-m from the project's own COCO-trained ONNX release | `pipelines/pose_estimation/models.py`, `pipelines/models/normal/detector_end2end.onnx` | Apache-2.0 |
| [OpenSim](https://github.com/opensim-org/opensim-core) | Musculoskeletal model scaling and inverse kinematics | `pipelines/kinematics/` | Apache-2.0 |

### Licensing note

Everything above is permissively licensed except one optional checkpoint, worth knowing
about before you distribute anything built on this repository.

**VideoPose3D checkpoint — CC BY-NC 4.0.** Not redistributable and not usable
commercially, so it is deliberately not bundled here; you download it yourself (see
[VideoPose3D lifting weights](#videopose3d-lifting-weights)). It is only required for
automatic calibration. Capture, pose estimation, board-based calibration, reconstruction
and kinematics never touch it, so for commercial use calibrate with the Object or
CheckerBoard targets instead, or supply your own calibration file.

None of this is legal advice; check with whoever owns the licensing decision for your
deployment.

