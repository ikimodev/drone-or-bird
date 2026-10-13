@echo off
title Push Updates to GitHub
cd /d "%~dp0"
echo ==================================================
echo Pushing Drone vs Bird C2 Updates to GitHub...
echo Repository: https://github.com/ikimodev/drone-or-bird
echo ==================================================
echo.
"C:\Users\krm12\.gemini\antigravity-ide\scratch\mingit\cmd\git.exe" push origin main
if %errorlevel% neq 0 (
    echo.
    echo ==================================================
    echo [!] Requires GitHub Authentication:
    echo Please authenticate via your GitHub account or Personal Access Token (PAT).
    echo ==================================================
) else (
    echo.
    echo ==================================================
    echo [SUCCESS] Successfully pushed all updates to GitHub!
    echo ==================================================
)
pause
