@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set CMD=%1
if "%CMD%"=="" goto :help
shift
set ARGS=
:collect
if "%~1"=="" goto :run
set ARGS=%ARGS% %1
shift
goto :collect
:run
if "%CMD%"=="test" %PY% -m pytest tests -q %ARGS% & exit /b
if "%CMD%"=="train" %PY% -m parkour.train %ARGS% & exit /b
if "%CMD%"=="sweep" %PY% scripts\sweep.py %ARGS% & exit /b
if "%CMD%"=="feasibility" %PY% -m parkour.feasibility %ARGS% & exit /b
if "%CMD%"=="report" %PY% -m parkour.report %ARGS% & exit /b
if "%CMD%"=="live" %PY% -m parkour.live %ARGS% & exit /b
if "%CMD%"=="offline" %PY% -m parkour.live --offline %ARGS% & exit /b
if "%CMD%"=="server" %PY% -m parkour.server run %ARGS% & exit /b
if "%CMD%"=="accept-eula" %PY% -m parkour.server accept-eula %ARGS% & exit /b
:help
echo Usage: run.cmd COMMAND [options]
echo   test          run all automated checks
echo   train         train a policy           (run.cmd train --run NAME)
echo   sweep         run a list of trainings  (run.cmd sweep configs\sweep_ablation.json)
echo   feasibility   recompute which jumps are possible
echo   report        tables and plots from finished runs
echo   offline       play held-out courses with the JS bot on Mineflayer's physics, no server
echo   live          play held-out courses in real Minecraft (needs accept-eula once)
echo   server        start the local parkour server by hand
echo   accept-eula   record your acceptance of the Minecraft EULA on this PC
exit /b 1
