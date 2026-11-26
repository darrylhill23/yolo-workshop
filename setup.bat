@echo off
cd /d "%~dp0"

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 goto failed

echo Installing CPU version of PyTorch...
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed

echo Installing remaining dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo Verifying installation...
venv\Scripts\python.exe -c "import torch, cv2, mediapipe, ultralytics; print('All packages imported successfully!')"
if errorlevel 1 goto failed

echo Installation complete!
call venv\Scripts\activate.bat
echo You are now ready to begin the workshop.
cmd /k
exit /b 0

:failed
echo Installation failed. Please contact your instructor.
cmd /k
exit /b 1
