import cv2
from ultralytics import YOLO
import os
import argparse
import sys
from pathlib import Path
import yaml
import torch




def ft(t: float):
    h=int(t/60/60)
    m=int((t-3600*h)/60)
    s=int(t-3600*h-60*m)
    f="{:02d}:{:02d}:{:02d}".format(h,m,s)
    return f

 
class Tracker: 
    def __init__(self): 
            self.weights='models/smokeroboflow_yolov8s_640_best.pt'
            self.conf=0.25
            self.imgsz=640
            self.half=False
            self.device='cpu'
            self.frame_id=0
            self.tracks=[]
            self.intervals=[]
            self.model=YOLO(self.weights)
            self.model.to(self.device)

    def track(self,cap,fps=25):  
        self.track_id=0
        self.intervals=[]

        while cap.isOpened():
            success, frame = cap.read()
            self.frame_id+=1
            if success:
                results = self.model.predict(frame,imgsz=self.imgsz,half=self.half,conf=self.conf)
                res = results[0] 
                boxes = res.boxes
                if (len(boxes)>0):
                    self.tracks.append(self.frame_id/fps)
            else:
               break 

    def calc_intervals(self,secs): 
        if len(self.tracks)==0: 
            return self.intervals
        n=len(self.tracks)
        i=0
        t0=self.tracks[i]
        segment=[ft(t0),ft(t0)]
        j=1
        while i<n-1 and j<n-1:
            tj=self.tracks[j]

            if tj-t0<=secs: 
                segment[1]=ft(tj)
                j+=1
                t0=tj
                if j==n-1:
                    self.intervals.append(segment)
                    break
                continue
               
            if tj-t0>secs: 
                segment[1]=ft(t0)
                self.intervals.append(segment)
                segment=[ft(tj),ft(tj)]
                i=j
                j+=1
                if j==n-1:
                    self.intervals.append(segment)
                    break
                t0=tj
