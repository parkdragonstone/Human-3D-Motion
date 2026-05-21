@echo off
setlocal

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_exe_gpu.ps1"
exit /b %ERRORLEVEL%
