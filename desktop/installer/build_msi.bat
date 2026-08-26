@echo off
REM Run this on Windows AFTER build.bat has produced desktop\dist\PNRC-OCR.exe.
REM Requires WiX Toolset v5 CLI: dotnet tool install --global wix

cd /d "%~dp0"

if not exist "..\dist\PNRC-OCR.exe" (
    echo PNRC-OCR.exe not found in desktop\dist\. Run build.bat first.
    pause
    exit /b 1
)

wix eula accept wix7
wix build product.wxs -o PNRC-OCR.msi
if errorlevel 1 (
    echo.
    echo wix build failed - see error above.
    pause
    exit /b 1
)

echo.
echo Done. Find PNRC-OCR.msi in desktop\installer\
pause
