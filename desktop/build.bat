@echo off
REM Run this on Windows to build PNRC-OCR.exe (PyInstaller can't cross-build from Linux/Mac).

cd /d "%~dp0"

python -m venv build_env
call build_env\Scripts\activate.bat

pip install -r requirements.txt

pyinstaller --onefile --windowed --name "PNRC-OCR" --add-data "ui.html;." app.py

echo.
echo Done. Find PNRC-OCR.exe in desktop\dist\
pause
