from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import time
import groundingdino.datasets.transforms as T
import numpy as np
import cv2
from PIL import Image
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




class Gracker: 
    def __init__(self): 
        self.cpath='/home/appuser/GroundingDINO_SwinT_OGC.py'
        self.cpoint='/home/appuser/groundingdino_swint_ogc.pth'
        self.text_threshold=0.25
        self.box_threshold=0.35
        self.conf=[self.text_threshold,self.box_threshold]
        self.text_prompt='cigarette'
        self.frame_id=0
        self.tracks=[]
        self.intervals=[]
        print(self.text_threshold)
        print(self.box_threshold)
        self.model=load_model(self.cpath, self.cpoint)



    def track(self,cap,fps=25):  
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.track_id=0
        self.intervals=[]

        while cap.isOpened():
            success, frame = cap.read()
            self.frame_id+=1
            if success:
                im = Image.fromarray(frame)
                image, _ = transform(im, None)
                start=time.time()
                print('text thre',self.text_threshold)
                print('box thre',self.box_threshold)
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image,
                    caption=self.text_prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )
                stop=time.time()
                print('frame:',self.frame_id)
                if (len(boxes)>0):
                    score = logits.cpu().numpy().tolist()
                    conf=score[0]
                    self.tracks.append(self.frame_id/fps)
                    print('inference time ms:',stop-start)
                    print('time:',self.tracks[-1], 'found:',phrases,score,boxes)
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
