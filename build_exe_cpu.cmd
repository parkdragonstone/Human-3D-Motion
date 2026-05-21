@echo off
setlocal

call "%~dp0scripts\build_exe_cpu.cmd"
exit /b %ERRORLEVEL%
