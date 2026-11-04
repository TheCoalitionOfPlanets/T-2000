@echo off
setlocal

REM The Windows launcher uses the self-contained high-VRAM deployment.
set "project_root=%~dp0"
cd /d "%project_root%"
if errorlevel 1 exit /b 1
set "force_build=false"
set "compose=docker compose -p t2000-high-vram -f compose.high-vram.yaml"

:parse_args
if "%~1"=="" goto check_docker
if "%~1"=="--build" (
    set "force_build=true"
    shift
    goto parse_args
)
if "%~1"=="--high-vram" (
    shift
    goto parse_args
)
if "%~1"=="--help" goto show_help
if "%~1"=="-h" goto show_help

echo Usage: %0 [--build] [--high-vram] 1>&2
exit /b 2

:show_help
echo Usage: %0 [--build] [--high-vram]
echo Start the self-contained high-VRAM T-2000 studio.
echo Use --build to refresh its image before starting.
exit /b 0

:check_docker
%compose% config --quiet
if errorlevel 1 exit /b 1

if "%force_build%"=="true" (
    call :build_image
    if errorlevel 1 exit /b 1
    goto start_studio
)

docker image inspect t2000:high-vram >nul 2>&1
if errorlevel 1 (
    call :build_image
    if errorlevel 1 exit /b 1
)

:start_studio
REM Initial model provisioning can take up to an hour.
%compose% up -d --no-build --pull never --wait --wait-timeout 3600 studio
if errorlevel 1 exit /b 1

set "address="
for /f "tokens=*" %%i in ('%compose% port studio 7861') do set "address=%%i"
if not defined address exit /b 1
echo T-2000 is ready at http://%address%
exit /b 0

:build_image
%compose% build studio
if errorlevel 1 exit /b 1
exit /b 0
