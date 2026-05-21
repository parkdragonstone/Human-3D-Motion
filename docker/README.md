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

Open the app:

```text
http://127.0.0.1:5000
```

Volumes mounted by default:

- `../recordings:/app/recordings`
- `../webapp_data:/app/webapp_data`
- `../pipelines/models:/app/pipelines/models:ro`

Download the `models` folder separately and place it at `pipelines/models` before running pose estimation.

GPU execution requires Docker with NVIDIA Container Toolkit.

The Docker build context is the repository root, so `.dockerignore` stays at the repository root.
