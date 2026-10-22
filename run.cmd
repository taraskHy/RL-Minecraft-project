@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" %*
set "projectExit=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %projectExit%
