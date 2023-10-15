import cv2
import os
from Quadro import Quadro
import argparse
import datetime

parser = argparse.ArgumentParser(description='Videos to images')
parser.add_argument('source_dir',nargs='?', help='film directory or parts', type=str, default='/usr/src/app/parts')
parser.add_argument('logfile', type=str, nargs='?', help="log file or log.txt", default='/usr/src/app/parts/log.txt')
parser.add_argument('weights', nargs='?', help='comma separated list of weights', type=str, default='/usr/src/app/models/smoke_y5m.pt,/usr/src/app/models/smoke_y5l.pt')
parser.add_argument('conf_thres', type=str, nargs='?', help="conf_thres or 0.05", default=0.05)
parser.add_argument('iou_thres', type=str, nargs='?', help="iou_thres or 0.45", default=0.45)

args = parser.parse_args()
print(args)

def ft(t: float):
    h=int(t/60/60)
    m=int((t-3600*h)/60)
    s=int(t-3600*h-60*m)
    f="{:02d}:{:02d}:{:02d}".format(h,m,s)
    return f

class QuadroSmoke(Quadro):
    def __init__(self,conf_path: str):
       super().__init__(conf_path)
       self.files=[]
       self.tracks=[]
       self.intervals=[]
       self.save_txt=True

    def save_labels(self,txt_path,frame,fps):
        fn_frm = os.path.basename(txt_path)
        fn,frm=fn_frm.split('_')
        if (len(self.files)>0 and fn==self.files[-1]):
            self.tracks[-1].append(int(frm)/fps)
        else:
            self.files.append(fn)
            self.tracks.append([int(frm)/fps])



    def calc_intervals(self,k,secs):
       tracks=self.tracks[k]
       intervals=[]
       if len(tracks)==0:
           return intervals

       n=len(tracks)
       i=0
       t0=tracks[i]
       segment=[ft(t0),ft(t0)]
       j=1
       while i<n-1 and j<n-1:
           tj=tracks[j]

           if tj-t0<=secs:
               segment[1]=ft(tj)
               #print('extend seg',tj,t0,segment)
               j+=1
               t0=tj
               if j==n-1:
                   #print("[",tj,"-",t0,"]")
                   intervals.append(segment)
                   return intervals
               continue

           if tj-t0>secs:
               segment[1]=ft(t0)
               #print('new seg',tj,t0,segment)
               intervals.append(segment)
               #print(segment)
               #print("[",tj,"-",t0,"]")
               segment=[ft(tj),ft(tj)]
               i=j
               j+=1
               if j==n-1:
                   intervals.append(segment)
                   return intervals
               t0=tj
       
    def print_intervals(self): 
        # print intervals to logfile

        now = datetime.datetime.now()
        print ("Current date and time : ")
        lf=open(logfile,"a")
        lf.write('start log: '+now.strftime("%Y-%m-%d %H:%M:%S")+'\n')
        for k,fn in enumerate(self.files): 
            ints=''.join(map(str, self.calc_intervals(k,2)))
            s=fn+' '+ints+'\n'
            lf.write(s)
        lf.close()       





source_dir=args.source_dir
logfile=args.logfile
home="/usr/src/app";
os.chdir(home)
q  = QuadroSmoke('quadro.yaml')
q.conf_tres = args.conf_thres
q.iou_tres = args.iou_thres
q.source_dir = args.source_dir
q.weights=args.weights.split(',')
print("Q weights:",q.weights)
q.get_source()
q.set_savedir()
q.load_model()
q.get_dataloader()
q.run_inference()
q.print_intervals()
