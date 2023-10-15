import argparse
import os
import platform
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import yaml

class Quadro:
    def __init__(self,cfg_yaml_path: str='quadro.yaml'):
        with open(cfg_yaml_path, errors='ignore') as f:
             opt = yaml.safe_load(f)
             print(opt)
        self.weights=opt['weights']
        self.source=opt['source']
        self.data=opt['data']
        self.imgsz=opt['imgsz']
        self.conf_thres=opt['conf_thres']
        self.iou_thres=opt['iou_thres']
        self.max_det=opt['max_det']
        self.device=opt['device']
        self.view_img=opt['view_img']
        self.save_txt=opt['save_txt']
        self.save_conf=opt['save_conf']
        self.save_crop=opt['save_crop']
        self.nosave=opt['nosave']
        self.classes=opt['classes']
        self.agnostic_nms=opt['agnostic_nms']
        self.augment=opt['augment']
        self.visualize=opt['visualize']
        self.update=opt['update']
        self.project=opt['project']
        self.name=opt['name']
        self.exist_ok=opt['exist_ok']
        self.line_thickness=opt['line_thickness']
        self.hide_labels=opt['hide_labels']    
        self.hide_conf=opt['hide_conf']
        self.half=opt['half']
        self.dnn=opt['dnn']
        self.vid_stride=opt['vid_stride']

    def get_source(self): 
        print('get_source')
        self.source = str(self.source)
        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.streams') or (self.is_url and not self.is_file)
        print('webcam:',self.webcam)    
        self.screenshot =     self.source.lower().startswith('screen')
        if self.is_url and self.is_file: 
            self.source = check_file(self.source)  # download 
     
    def set_savedir(self):
        # Directories
        # increment run (self.save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  
        (self.save_dir / 'labels' if self.save_txt else self.save_dir) #.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    def load_model(self): 
       self.device = select_device(self.device)
       self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
       self.stride,self.names,self.pt=self.model.stride, self.model.names, self.model.pt
       self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    # Dataloader
    def get_dataloader(self): 
        self.bs = 1  # batch_size
        if self.webcam: 
            self.view_img = check_imshow(warn=True)
            self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
            self.bs = len(self.dataset)
        elif self.screenshot:
            self.dataset = LoadScreenshots(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        else:
            self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)


    def save_labels(self,txt_path,frame,fps): 
        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
        with open(f'{txt_path}.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

    @smart_inference_mode()
    def run_inference(self): 
        vid_path, vid_writer = [None] * self.bs, [None] * self.bs

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in self.dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                self.visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred = self.model(im, augment=self.augment, visualize=self.visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    s += f'{frame}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
                    s += f'{frame}: '

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]

                            self.save_labels(txt_path,frame,fps)

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if self.view_img and not self.daemon:
                    print('view_img',self.view_img)
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if self.dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
#        if self.save_txt or self.save_img:
#            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
#            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
#        if self.update:
#            strip_optimizer(self.weights[0])  # update model (to fix SourceChangeWarning)

