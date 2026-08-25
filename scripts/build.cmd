@echo off
setlocal

rem Human3DMotion - Windows executable build
rem Usage: build.cmd [cpu|gpu]
rem   no argument -> auto-detect (GPU when nvidia-smi is available)

set "REPO_ROOT=%~dp0.."
set "ENV_NAME=human-3d-motion"
set "MODE=%~1"

if /i "%MODE%"=="" goto :detect_mode
if /i "%MODE%"=="cpu" goto :mode_ready
if /i "%MODE%"=="gpu" goto :mode_ready
if /i "%MODE%"=="auto" goto :detect_mode
echo [ERROR] Unknown mode "%MODE%". Use: build.cmd [cpu^|gpu]
exit /b 1

:detect_mode
where nvidia-smi >nul 2>&1
if errorlevel 1 (set "MODE=cpu") else (set "MODE=gpu")
echo [INFO] Auto-detected mode: %MODE%
goto :mode_ready

:mode_ready
pushd "%REPO_ROOT%" || exit /b 1

if not exist "pipelines\models\normal\rtmpose_end2end.onnx" goto :fail_models

call "%~dp0env_setup.cmd" %MODE%
if errorlevel 1 goto :fail_env

echo [INFO] Building frontend assets
call npm install
if errorlevel 1 goto :fail_frontend
call npm run build:ts
if errorlevel 1 goto :fail_frontend

echo [INFO] Creating the executable icon
call conda run -n %ENV_NAME% python packaging\create_app_icon.py
if errorlevel 1 goto :fail_icon

echo [INFO] Building the executable
call conda run -n %ENV_NAME% python -m pip install "pyinstaller>=6.0"
if errorlevel 1 goto :fail_pyinstaller
call conda run -n %ENV_NAME% python -m PyInstaller --noconfirm --clean packaging\Human3DMotion.spec
if errorlevel 1 goto :fail_pyinstaller

popd
echo.
echo [DONE] Executable build complete (%MODE%):
echo        %REPO_ROOT%\dist\Human3DMotion\Human3DMotion.exe
exit /b 0

:fail_models
popd
echo [ERROR] Pose model files are missing.
echo         Place the models folder at pipelines\models before building.
exit /b 1

:fail_env
popd
echo [ERROR] Environment setup failed.
exit /b 1

:fail_frontend
popd
echo [ERROR] Building the frontend assets failed.
exit /b 1

:fail_icon
popd
echo [ERROR] Creating the executable icon failed.
exit /b 1

:fail_pyinstaller
popd
echo [ERROR] PyInstaller build failed.
exit /b 1
