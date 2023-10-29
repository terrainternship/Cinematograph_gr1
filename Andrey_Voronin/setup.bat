@echo off
if exist SD_venv (
  echo Setup environment already done
  pause
) else (
  echo Start setting up the environment. This may take some time.
  pause
  python -m venv SD_venv\
  SD_venv\Scripts\activate.bat
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install opencv-python-headless
  pip install ultralytics
  SD_venv\Scripts\deactivate.bat
)
