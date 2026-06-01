$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$environmentName = "human-3d-motion-exe"

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

function New-CpuBuildEnvironment {
    conda env create -n $environmentName -f environment-cpu.yml
    Assert-LastCommandSucceeded "Creating CPU executable Conda environment"
}

Push-Location $repoRoot
try {
    if (-not (Test-CondaEnvironment $environmentName)) {
        New-CpuBuildEnvironment
    }

    if (-not (Test-Path "pipelines\models\normal\rtmpose_end2end.onnx")) {
        throw "Pose model files are missing. Place the models folder at pipelines\models before building the executable."
    }

    conda run -n $environmentName python -c "import openvino; print('OpenVINO:', openvino.__version__)"
    Assert-LastCommandSucceeded "Verifying OpenVINO"
    conda run -n $environmentName python -c "from pipelines.poseEstimation import setup_backend_device; resolved = setup_backend_device('auto', 'auto'); assert resolved == ('openvino', 'cpu'), resolved; print('Pose backend/device:', resolved)"
    Assert-LastCommandSucceeded "Verifying CPU pose backend auto-detection"

    npm.cmd install
    Assert-LastCommandSucceeded "Installing frontend dependencies"
    npm.cmd run build:ts
    Assert-LastCommandSucceeded "Building frontend assets"

    conda run -n $environmentName python packaging\create_app_icon.py
    Assert-LastCommandSucceeded "Creating executable icon"

    conda run -n $environmentName python -m pip install "pyinstaller>=6.0"
    Assert-LastCommandSucceeded "Installing PyInstaller"
    conda run -n $environmentName python -m PyInstaller --noconfirm --clean packaging\Human3DMotion.spec
    Assert-LastCommandSucceeded "Building 3DHumanMotion executable"

    Write-Host "Executable build complete: $repoRoot\dist\Human3DMotion\Human3DMotion.exe"
} finally {
    Pop-Location
}
