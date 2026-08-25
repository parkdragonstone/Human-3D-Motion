@echo off
setlocal

rem Human3DMotion - Windows environment setup (no build)
rem Usage: env_setup.cmd [cpu|gpu]
rem   no argument -> auto-detect (GPU when nvidia-smi is available)

set "REPO_ROOT=%~dp0.."
set "ENV_NAME=human-3d-motion"
set "MODE=%~1"

if /i "%MODE%"=="" goto :detect_mode
if /i "%MODE%"=="cpu" goto :mode_ready
if /i "%MODE%"=="gpu" goto :mode_ready
if /i "%MODE%"=="auto" goto :detect_mode
echo [ERROR] Unknown mode "%MODE%". Use: env_setup.cmd [cpu^|gpu]
exit /b 1

:detect_mode
where nvidia-smi >nul 2>&1
if errorlevel 1 (set "MODE=cpu") else (set "MODE=gpu")
echo [INFO] Auto-detected mode: %MODE%
goto :mode_ready

:mode_ready
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda was not found on PATH. Run this script from an Anaconda/Miniconda prompt.
    exit /b 1
)

pushd "%REPO_ROOT%" || exit /b 1

echo [INFO] Preparing Conda environment "%ENV_NAME%" (%MODE%)
call conda env list | findstr /r /c:"^%ENV_NAME% " >nul
if not errorlevel 1 goto :env_exists

call conda env create -n %ENV_NAME% -f environment-%MODE%.yml
if errorlevel 1 goto :fail_create
if /i "%MODE%"=="cpu" goto :verify

call conda run -n %ENV_NAME% python -m pip uninstall -y onnxruntime onnxruntime-gpu
if errorlevel 1 goto :fail_ort
call conda run -n %ENV_NAME% python -m pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.26.0
if errorlevel 1 goto :fail_ort
goto :verify

:env_exists
echo [INFO] Conda environment "%ENV_NAME%" already exists. Reusing it.
goto :verify

:verify
if /i "%MODE%"=="gpu" goto :verify_gpu

call conda run -n %ENV_NAME% python -c "import openvino; print('OpenVINO:', openvino.__version__)"
if errorlevel 1 goto :fail_verify
call conda run -n %ENV_NAME% python -c "from pipelines.poseEstimation import setup_backend_device; resolved = setup_backend_device('auto', 'auto'); assert resolved == ('openvino', 'cpu'), resolved; print('Pose backend/device:', resolved)"
if errorlevel 1 goto :fail_verify
goto :install_command

:verify_gpu
call conda run -n %ENV_NAME% python -c "import torch; assert torch.cuda.is_available(), f'Torch CUDA unavailable: {torch.__version__}, CUDA={torch.version.cuda}'; print('Torch CUDA:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
if errorlevel 1 goto :fail_verify
call conda run -n %ENV_NAME% python -c "import onnxruntime as ort; providers = ort.get_available_providers(); assert 'CUDAExecutionProvider' in providers, providers; print('ONNX Runtime providers:', providers)"
if errorlevel 1 goto :fail_verify
call conda run -n %ENV_NAME% python -c "from pipelines.poseEstimation import setup_backend_device; resolved = setup_backend_device('auto', 'auto'); assert resolved == ('onnxruntime', 'cuda'), resolved; print('Pose backend/device:', resolved)"
if errorlevel 1 goto :fail_verify
goto :install_command

:install_command
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_h3dm_command.ps1" -EnvironmentName %ENV_NAME%
if errorlevel 1 goto :fail_command

popd
echo.
echo [DONE] Environment "%ENV_NAME%" (%MODE%) is ready.
echo        Activate it with: conda activate %ENV_NAME%
echo        Start the app with: h3dm
exit /b 0

:fail_create
popd
echo [ERROR] Creating the Conda environment failed.
exit /b 1

:fail_ort
popd
echo [ERROR] Installing onnxruntime-gpu failed.
exit /b 1

:fail_verify
popd
echo [ERROR] Environment verification failed.
exit /b 1

:fail_command
popd
echo [ERROR] Installing the h3dm command failed.
exit /b 1
