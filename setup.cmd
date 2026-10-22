@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" setup
set "projectExit=%ERRORLEVEL%"
if not "%projectExit%"=="0" echo Setup did not finish. Read the error above, fix it, and rerun setup.cmd.
pause
exit /b %projectExit%
