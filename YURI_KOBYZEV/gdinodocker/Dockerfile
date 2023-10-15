FROM kobyzevyuri/groundino:latest

# Arguments to build Docker Image using CUDA

WORKDIR /home/appuser
RUN mkdir tracker
COPY gr_track.py /home/appuser/
COPY tracker/Gracker.py /home/appuser/tracker/
COPY GroundingDINO_SwinT_OGC.py /home/appuser/
COPY groundingdino_swint_ogc.pth /home/appuser/
COPY groundino_yolo.py /home/appuser/
