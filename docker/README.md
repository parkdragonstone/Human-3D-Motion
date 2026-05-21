# Docker

Run these commands from the repository root.

Build and run the CPU version:

```bash
docker compose -f docker/docker-compose.yml --profile cpu up --build
```

Build and run the GPU version:

```bash
docker compose -f docker/docker-compose.yml --profile gpu up --build
```

For phone capture over the LAN from Windows PowerShell, use the launch scripts:

```powershell
.\docker\run_cpu.ps1
.\docker\run_gpu.ps1
```

The scripts detect active host LAN IPv4 candidates, set HTTPS and
`BASEBALL_MOTION_PUBLIC_URL`, print the selected LAN URL, and start Compose.
When the detected address is not the network the phone can reach, override it:

```powershell
.\docker\run_gpu.ps1 -HostIp 192.168.0.10
```

The Flask `Running on ...` lines still show bind addresses inside Docker such as
`127.0.0.1` and `172.x.x.x`. The app prints `Baseball Motion public URL: ...`
separately when the script-provided LAN URL is active.

When Docker runs inside WSL, use the shell launch scripts from WSL:

```bash
bash docker/run_cpu.sh
bash docker/run_gpu.sh
```

The WSL scripts call `powershell.exe` to detect the Windows host LAN IPv4
candidates and open a Windows folder chooser before Compose starts. The chosen
folder is mounted into the container as `/app/recordings`, which becomes the
Capture, Calibration, and Analysis root. Override the selected address when
needed:

```bash
bash docker/run_gpu.sh --host-ip 192.168.0.10
```

When the folder chooser cannot open, pass an existing WSL-visible folder path:

```bash
bash docker/run_gpu.sh --storage-dir /mnt/c/Users/USER/Desktop/PYS/BaseballMotion/recordings
```

The in-app `Select Path` button cannot mount a new arbitrary Windows folder
after the WSL Docker container is already running. Restart with the WSL script
to choose a different Windows storage folder.

WSL LAN access is separate from the app public URL. Depending on the WSL
network mode, another device may still need Windows mirrored networking or a
Windows port proxy and firewall rule for port `5000`.

Open the app:

```text
http://127.0.0.1:5000
```

For another device on the same network, use the Docker host LAN address:

```text
http://<docker-host-lan-ip>:5000
```

Use the Docker host LAN address, not the container address printed as
`172.x.x.x` in the Flask log.

Phone camera access from another device requires a secure browser context. The
scripts above set this automatically. To start Compose manually, turn on the
self-signed development HTTPS server and set the public URL first:

```powershell
$env:BASEBALL_MOTION_HTTPS="1"
$env:BASEBALL_MOTION_PUBLIC_URL="https://<docker-host-lan-ip>:5000"
docker compose -f docker/docker-compose.yml --profile cpu up --build
```

Continue through the browser certificate warning for local HTTPS use. If another
device cannot open the LAN page, allow inbound TCP port `5000` on the Docker
host firewall.

Volumes mounted by default:

- `../recordings:/app/recordings`
- `../webapp_data:/app/webapp_data`
- `../pipelines/models:/app/pipelines/models:ro`

Download the `models` folder separately and place it at `pipelines/models` before running pose estimation.

GPU execution requires Docker with NVIDIA Container Toolkit.

The Docker build context is the repository root, so `.dockerignore` stays at the repository root.
