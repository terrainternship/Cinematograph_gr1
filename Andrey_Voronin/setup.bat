@echo off
if exist SD_venv (
  echo Setup environment already done
  pause
) else (
  echo Start setting up the environment. This may take some time.
  pause
  python -m venv SD_venv\
  SD_venv\Scripts\activate.bat
  pip3 install opencv-python-headless
  pip3 install ultralytics
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)