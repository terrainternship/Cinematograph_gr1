import cv2
import os
from tracker.Tracker import Tracker
import argparse
parser = argparse.ArgumentParser(description='Videos to images')
#/usr/src/app/models/
parser.add_argument('source_dir',nargs='?', help='film directory or parts', type=str, default='/usr/src/ultralytics/parts')
parser.add_argument('logfile', type=str, nargs='?', help="log file or log.txt", default='/usr/src/ultralytics/parts/log.txt')
parser.add_argument('imgsz', type=str, nargs='?', help=" image size 640", default=640)
parser.add_argument('conf', type=str, nargs='?', help=" or 0.25", default=0.25)
parser.add_argument('half', type=str, nargs='?', help="half=False", default=False)
parser.add_argument('device', type=str, nargs='?', help="cpu", default=0)

args = parser.parse_args()
print(args)



source_dir=args.source_dir
logfile=args.logfile
home="/usr/src/ultralytics";
os.chdir(home)
q  = Tracker()
q.device = args.device
q.conf = args.conf
q.half = args.half
q.source_dir = args.source_dir

log = open(logfile,'a')

for root, dirs, files in os.walk(source_dir, topdown=False):
   for name in files:
      source = os.path.join(root, name)
      print('file:',name)
      basename,mp4=name.split(".")
      if mp4!='mp4' and mp4!='avi' and mp4!='mkv':
          print('name not mp4 file:',name)
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
