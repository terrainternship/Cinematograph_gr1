import warnings
from io import BytesIO
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import time
import numpy as np
import cv2
from PIL import Image
import os
import argparse
import sys
from pathlib import Path
import yaml
import torch

import argparse
parser = argparse.ArgumentParser(description='Videos to images')
parser.add_argument('image_path',nargs='?', help='image directory or film', type=str, default='/images')
parser.add_argument('label_path', type=str, nargs='?', help="log file or log.txt", default='/images/log.txt')
parser.add_argument('text_prompt', type=str, nargs='?', help="text prompt or cigarette", default='cigarette')
parser.add_argument('box_threshold', type=str, nargs='?', help="box_threshold or 0.35", default=0.35)
parser.add_argument('text_threshold', type=str, nargs='?', help="text_threshold or 0.25", default=0.25)
args = parser.parse_args()
print(args)
home="/home/appuser";
os.chdir(home)


CONFIG_PATH='GroundingDINO_SwinT_OGC.py'
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"
model = load_model(CONFIG_PATH, WEIGHTS_PATH)

TEXT_PROMPT = "cigarette"
BOX_TRESHOLD = args.box_threshold
TEXT_TRESHOLD = args.text_threshold
TEXT_PROMPT = args.text_prompt
image_path = args.image_path
label_path = args.label_path

def main():
    for image in os.listdir(image_path):
      base=Path(image).stem
      image_source, image_pil = load_image(image_path+'/'+image)
      boxes, logits, phrases = predict(
          model=model,
          image=image_pil,
          caption=TEXT_PROMPT,
          box_threshold=BOX_TRESHOLD,
          text_threshold=TEXT_TRESHOLD
      )
      label=label_path+'/'+base+'.txt'
      txt = open(label, "w")
      print('file:',label)
      print(boxes)
      if len(boxes) == 0:
          print(f"No objects of the '{TEXT_PROMPT}' prompt detected in the image.")
          txt.close()
      else:
          # Save the boxes
          i=0
          for box, logit in zip(boxes,logits):
            i+=1
            x_min, y_min, x_max, y_max = box
            confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
            sl = f"Logit {i+1}: {confidence_score}\n"
            s=f"0 {x_min} {y_min} {x_max} {y_max}\n"
            print(sl,s)
            txt.write(s)
          print(f"found",i)
          txt.close()

if __name__ == "__main__":
    main()

