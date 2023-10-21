@echo off


set venv_root_dir="SD_venv"


call %venv_root_dir%\Scripts\activate.bat
python main.py 
call %venv_root_dir%\Scripts\deactivate.bat

exit /B 1