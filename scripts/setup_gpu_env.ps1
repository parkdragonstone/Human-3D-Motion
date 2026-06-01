$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$environmentName = "human-3d-motion"

function Assert-LastCommandSucceeded {
    param([string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

Push-Location $repoRoot
try {
    $environmentList = conda env list
    Assert-LastCommandSucceeded "Listing Conda environments"
    if ($environmentList | Select-String -Pattern "^\s*$environmentName\s+") {
        throw "Conda environment '$environmentName' already exists. Remove or rename it before creating a fresh GPU environment."
    }

    conda env create -f environment-gpu.yml
    Assert-LastCommandSucceeded "Creating GPU Conda environment"

    conda run -n $environmentName python -m pip uninstall -y onnxruntime onnxruntime-gpu
    Assert-LastCommandSucceeded "Removing conflicting ONNX Runtime packages"
    conda run -n $environmentName python -m pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.26.0
    Assert-LastCommandSucceeded "Installing ONNX Runtime GPU"

    conda run -n $environmentName python -c "import torch; assert torch.cuda.is_available(), f'Torch CUDA unavailable: {torch.__version__}, CUDA={torch.version.cuda}'; print('Torch CUDA:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
    Assert-LastCommandSucceeded "Verifying Torch CUDA"
    conda run -n $environmentName python -c "import onnxruntime as ort; providers = ort.get_available_providers(); assert 'CUDAExecutionProvider' in providers, providers; print('ONNX Runtime providers:', providers)"
    Assert-LastCommandSucceeded "Verifying ONNX Runtime CUDA provider"
    conda run -n $environmentName python -c "from pipelines.poseEstimation import setup_backend_device; resolved = setup_backend_device('auto', 'auto'); assert resolved == ('onnxruntime', 'cuda'), resolved; print('Pose backend/device:', resolved)"
    Assert-LastCommandSucceeded "Verifying pose backend auto-detection"

    & (Join-Path $PSScriptRoot "install_h3dm_command.ps1") -EnvironmentName $environmentName
    Assert-LastCommandSucceeded "Installing h3dm command"
} finally {
    Pop-Location
}
