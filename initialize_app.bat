@echo off
setlocal

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 exit /b 1

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist Vision_Based_Fastener_Inspection.spec del /f /q Vision_Based_Fastener_Inspection.spec

pyinstaller --onefile --windowed ^
    --name Vision_Based_Fastener_Inspection ^
    --icon "ui\assets\icon.ico" ^
    --add-data "ui\assets\logo.png;ui\assets" ^
    backend\launcher.py

if errorlevel 1 exit /b 1

move /Y "dist\Vision_Based_Fastener_Inspection.exe" ".\Vision_Based_Fastener_Inspection.exe"

endlocal
