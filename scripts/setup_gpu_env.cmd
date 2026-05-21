@echo off
setlocal

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_gpu_env.ps1"
exit /b %ERRORLEVEL%
