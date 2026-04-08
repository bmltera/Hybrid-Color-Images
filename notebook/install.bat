@echo off
REM Create venv folder next to the notebook folder
cd /d %~dp0
python -m venv ..\venv
call ..\venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
echo.
echo Setup complete.
echo To activate the environment in the future, run:
echo     ..\venv\Scripts\activate
