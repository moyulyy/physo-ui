@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title PhySO Studio
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "ENV_NAME=physo-ui"

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda was not found in PATH.
    echo         Install Miniconda/Anaconda or run with a normal Python:
    echo             python -m physo_ui
    pause
    exit /b 1
)

rem --- Locate the environment without relying on `conda init cmd.exe` ---
set "PYTHON_EXE="
for /f "delims=" %%B in ('conda info --base 2^>nul') do set "CONDA_BASE=%%B"
if defined CONDA_BASE (
    if exist "%CONDA_BASE%\envs\%ENV_NAME%\python.exe" set "PYTHON_EXE=%CONDA_BASE%\envs\%ENV_NAME%\python.exe"
)
if not defined PYTHON_EXE (
    if exist "%USERPROFILE%\.conda\envs\%ENV_NAME%\python.exe" set "PYTHON_EXE=%USERPROFILE%\.conda\envs\%ENV_NAME%\python.exe"
)

if not defined PYTHON_EXE (
    echo [ERROR] Conda environment "%ENV_NAME%" was not found.
    echo.
    echo Create it first:
    echo     conda create -n %ENV_NAME% python=3.11 -y -c conda-forge
    echo     conda activate %ENV_NAME%
    echo     pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem --- Quick dependency check ---
"%PYTHON_EXE%" -c "import PySide6, torch, physo" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] The environment is missing dependencies.
    echo.
    echo Run:
    echo     "%PYTHON_EXE%" -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Starting PhySO Studio with %PYTHON_EXE%
"%PYTHON_EXE%" -m physo_ui %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo PhySO Studio exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
