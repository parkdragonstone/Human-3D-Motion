$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$environmentName = "baseball-motion-exe"

function Assert-LastCommandSucceeded {
    param([string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Test-CondaEnvironment {
    param([string]$Name)

    $environmentList = conda env list
    Assert-LastCommandSucceeded "Listing Conda environments"
    return [bool]($environmentList | Select-String -Pattern "^\s*$Name\s+")
}

function New-GpuBuildEnvironment {
    conda env create -n $environmentName -f environment-gpu.yml
    Assert-LastCommandSucceeded "Creating GPU executable Conda environment"

    conda run -n $environmentName python -m pip uninstall -y onnxruntime onnxruntime-gpu
    Assert-LastCommandSucceeded "Removing conflicting ONNX Runtime packages"
    conda run -n $environmentName python -m pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.26.0
    Assert-LastCommandSucceeded "Installing ONNX Runtime GPU"
}

Push-Location $repoRoot
try {
    if (-not (Test-CondaEnvironment $environmentName)) {
        New-GpuBuildEnvironment
    }

    if (-not (Test-Path "pipelines\models\normal\rtmpose_end2end.onnx")) {
        throw "Pose model files are missing. Place the models folder at pipelines\models before building the executable."
    }

    conda run -n $environmentName python -c "import torch; assert torch.cuda.is_available(), f'Torch CUDA unavailable: {torch.__version__}, CUDA={torch.version.cuda}'; print('Torch CUDA:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
    Assert-LastCommandSucceeded "Verifying Torch CUDA"
    conda run -n $environmentName python -c "import onnxruntime as ort; providers = ort.get_available_providers(); assert 'CUDAExecutionProvider' in providers, providers; print('ONNX Runtime providers:', providers)"
    Assert-LastCommandSucceeded "Verifying ONNX Runtime CUDA provider"
    conda run -n $environmentName python -c "from pipelines.poseEstimation import setup_backend_device; resolved = setup_backend_device('auto', 'auto'); assert resolved == ('onnxruntime', 'cuda'), resolved; print('Pose backend/device:', resolved)"
    Assert-LastCommandSucceeded "Verifying pose backend auto-detection"

    npm.cmd install
    Assert-LastCommandSucceeded "Installing frontend dependencies"
    npm.cmd run build:ts
    Assert-LastCommandSucceeded "Building frontend assets"

    conda run -n $environmentName python packaging\create_app_icon.py
    Assert-LastCommandSucceeded "Creating executable icon"

    conda run -n $environmentName python -m pip install "pyinstaller>=6.0"
    Assert-LastCommandSucceeded "Installing PyInstaller"
    conda run -n $environmentName python -m PyInstaller --noconfirm --clean packaging\BaseballMotion.spec
    Assert-LastCommandSucceeded "Building BaseballMotion executable"

    Write-Host "Executable build complete: $repoRoot\dist\BaseballMotion\BaseballMotion.exe"
} finally {
    Pop-Location
}
