#sudo docker run -it --gpus all -v /vol/src/mytest/parts/:/usr/src/ultralytics/parts smokey8:latest bash
sudo docker run -it --gpus all -v /vol/src/mytest/parts:/usr/src/ultralytics/parts smokey8:latest python y8_track.py
