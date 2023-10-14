import cv2
import os
from tracker.Gracker import Gracker
import argparse
parser = argparse.ArgumentParser(description='Videos to images')
parser.add_argument('source_dir',nargs='?', help='film directory or parts', type=str, default='/home/appuser/films')
parser.add_argument('logfile', type=str, nargs='?', help="log file or log.txt", default='/home/appuser/films/log.txt')
parser.add_argument('text_prompt', type=str, nargs='?', help="text prompt or cigarette", default='cigarette')
parser.add_argument('box_threshold', type=str, nargs='?', help="box_threshold or 0.35", default=0.35)
parser.add_argument('text_threshold', type=str, nargs='?', help="text_threshold or 0.25", default=0.25)

args = parser.parse_args()
print(args)



source_dir=args.source_dir
logfile=args.logfile
home="/home/appuser";
os.chdir(home)
q  = Gracker()
q.box_treshold = args.box_threshold
q.text_treshold = args.text_threshold
q.text_prompt = args.text_prompt
q.source_dir = args.source_dir

log = open(logfile,'a')

for root, dirs, files in os.walk(source_dir, topdown=False):
   for name in files:
      source = os.path.join(root, name)
      print('file:',name)
      basename,mp4=name.split(".")
      if mp4!='mp4' and mp4!='avi' and mp4!='mkv':
          print('name not mp4 file continue...:',name)
          continue
      cap = cv2.VideoCapture(source)
      fps=cap.get(cv2.CAP_PROP_FPS)
      print('FPS:',fps)
      mytrack = q.track(cap,fps)
      cap.release()
      q.calc_intervals(2)
      ints=''.join(map(str, q.intervals))
      s=basename+' '+ints+'\n'
      log.write(s)
      q.tracks=[]
      q.intervals=[]
