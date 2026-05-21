@echo off
setlocal

call "%~dp0scripts\build_exe_gpu.cmd"
exit /b %ERRORLEVEL%
