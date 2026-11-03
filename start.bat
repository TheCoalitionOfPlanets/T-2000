@echo off
setlocal enabledelayedexpansion

REM Get project root directory
set "project_root=%~dp0"
cd /d "%project_root%"

set force_build=false

REM Parse command-line arguments
if "%1"=="" goto check_docker
if "%1"=="--build" (
    set force_build=true
    goto check_docker
)
if "%1"=="--help" goto show_help
if "%1"=="-h" goto show_help

echo Usage: %0 [--build] 1>&2
exit /b 2

:show_help
echo Usage: %0 [--build]
echo Start T-2000 using its installed image or local recovery archive.
echo Use --build to refresh the image from the canonical Dockerfile.
exit /b 0

:check_docker
docker compose config --quiet
if errorlevel 1 exit /b 1

if "%force_build%"=="true" (
    call :build_image
    goto start_studio
)

docker image inspect t2000:gpu >nul 2>&1
if errorlevel 1 (
    set "recovery_archive=%project_root%.cache\t2000-gpu.tar"
    if exist "!recovery_archive!" (
        docker image load -i "!recovery_archive!"
        docker image inspect t2000:gpu >nul 2>&1
        if errorlevel 1 (
            echo The recovery archive did not contain the expected t2000:gpu image. 1>&2
            exit /b 1
        )
    ) else (
        call :build_image
    )
)

:start_studio
docker compose up -d --no-build --pull never --wait --wait-timeout 180 studio
if errorlevel 1 exit /b 1

for /f "tokens=*" %%i in ('docker compose port studio 7861') do set address=%%i
echo T-2000 is ready at http://%address%
exit /b 0

:build_image
REM Check available disk space (in KB)
REM Note: This is a simplified version - Windows disk space checking is more complex
REM For production, consider using wmic or PowerShell
docker compose build studio
if errorlevel 1 exit /b 1
exit /b 0
