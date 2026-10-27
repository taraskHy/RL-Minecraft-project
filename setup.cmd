@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv || goto :error
)
.venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 || goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
rem Portable Node, Java, pnpm and the official Minecraft 1.21.4 server jar, inside .local (checksums verified).
.venv\Scripts\python.exe -c "from parkour.runtime import setup_runtime; setup_runtime()" || goto :error
echo.
echo Setup finished. Next: run.cmd test
exit /b 0
:error
echo Setup failed.
exit /b 1
