#!/usr/bin/env python
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    plot_one_box, strip_optimizer, set_logging, increment_dir
from utils.torch_utils import select_device, load_classifier, time_synchronized


class Detect():
    def __init__(self, opt):
        self.save_dir, self.source, self.weights, self.view_img, self.save_txt, self.imgsz = \
            Path(
                opt.save_dir), opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

        if self.save_dir == Path('runs/detect'):  # if default
            self.save_dir.mkdir(parents=True, exist_ok=True)  # make base
            self.save_dir = Path(increment_dir(
                self.save_dir / 'exp', opt.name))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True,
                                                                             exist_ok=True)  # make new dir

        set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'

        # Load model
        self.model = attempt_load(
            self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(
            self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            self.modelc.to(self.device).eval()

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        self.save_img = True
        self.dataset = LoadImages(self.source, img_size=self.imgsz)

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(len(self.names))]

    def detect(self):
        detections = {}
        count = 1
        t0 = time.time()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz),
                          device=self.device)  # init img
        # run once
        _ = self.model(
            img.half() if self.half else img) if self.device.type != 'cpu' else None
        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=opt.augment)[0]

            detections_pred = []
            detections_count= {}

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = Path(path), '', im0s

                save_path = str(self.save_dir / p.name)
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('_%g' %
                                                                     self.dataset.frame if self.dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += '%g %ss, ' % (n, self.names[int(c)])
                        detections_count[self.names[int(c)]] =  n.item()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                              ) / gn).view(-1).tolist()  # normalized xywh
                            # label format
                            line = (
                                cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line) + '\n') % line)
                        
                        label = '"%s" %.2f' % (self.names[int(cls)], conf)
                        detections_pred.append((label,xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item()))
                        


                        if self.save_img or self.view_img:  # Add bbox to image
                            plot_one_box(xyxy, im0, label=label,
                                         color=self.colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                #print(detections_pred)
                #print(detections_count)
                detections[p.name] =[detections_pred, detections_count]
                
                # Stream results
                if self.view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if self.dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

        if self.save_txt or self.save_img:
            print('Results saved to %s' % self.save_dir)

        print('Done. (%.3fs)' % (time.time() - t0))
        return(detections)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov5s.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str,
                        default='runs/detect', help='directory to save results')
    parser.add_argument(
        '--name', default='', help='name to append to --save-dir: i.e. runs/{N} -> runs/{N}_{name}')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    opt = parser.parse_args()
    print(opt)

    Detector = Detect(opt=opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detections=Detector.detect()
                strip_optimizer(opt.weights)
        else:
            detections=Detector.detect()
    
    print(detections)
