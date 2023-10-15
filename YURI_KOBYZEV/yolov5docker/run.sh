#sudo docker run -it --gpus all -v /vol/src/mytest/parts/:/usr/src/app/parts smokey5:latest bash
sudo docker run -it --gpus all -v /vol/src/mytest/parts:/usr/src/app/parts smokey5:latest python y5_track.py /usr/src/app/parts /usr/src/app/parts/log.txt "/usr/src/app/models/smoke_y5l.pt"
