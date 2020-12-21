#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_dir, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import csv
import sys
import os

parent_path = '/'
dir = '/enigma/local_storage/result'
if os.path.exists(dir):
    print("")
else:
    os.makedirs(dir)
webcam = False 
args = str(sys.argv)
print(args)
input_file = sys.argv[1]
output_prediction = sys.argv[2]
output_count = sys.argv[3]
files = []
save_img = True
view_img = False
with open(input_file) as input_file:
    input = csv.reader(input_file, delimiter=',')
    line_count = 0
    for row in input:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            f = row[0]
            if f[-1] == ";":
                f = f[:-1]
            f = "/enigma/datasets/HA-Sample/" + f
            files.append(f)
print(files)  

#########################################################
source = files
# add weights file
weights = "submission/weights/custom/best.pt"
# add image size
imgsz = 640
conf_thresh = 0.25  
save_txt = False
dev = ''
augment = True
# Directories
save_dir = Path(increment_dir(Path("test/test2/"))) # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Initialize
set_logging()
device = select_device(dev)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

# Set Dataloader
vid_path, vid_writer = None, None
if webcam:
    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz)
else:
    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

####################################

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
detections = {}
count = 1
t0 = time.time()
img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

# Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]


    detections_pred = []
    detections_count= {}

    # Apply NMS
    # pred = non_max_suppression(pred, conf_thresh, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    pred = non_max_suppression(pred, conf_thresh)

    t2 = time_synchronized()

        
        # Process detections
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
        else:
            p, s, im0 = Path(path), '', im0s

        save_path = str(save_dir / p.name)
        txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
        # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
                detections_count[names[int(c)]] =  n.item()
            view_img = False
            # Write results
            save_conf = False
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    detections_pred.append((label,xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item()))

            # Print time (inference + NMS)
        # print('%sDone. (%.3fs)' % (s, t2 - t1))
        detections[path] =[detections_pred, detections_count]
            # Stream results
        
        if view_img:
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

            # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)

if save_txt or save_img:
    print('Results saved to %s' % save_dir)

# print('Done. (%.3fs)' % (time.time() - t0))

####################################

with open(output_prediction, mode = 'w', newline='') as predictions:
    prediction_writer = csv.writer(predictions, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    prediction_writer.writerow(['ImagePath', 'PredictionString (class prediction score xmin ymin xmax ymax;)'])
    for key in detections:
        t = detections[key][0]
        strr = ''
        for item in t:
            for x in range(len(item)):
                i = item[x]
                i = str(i)
                if x == 0:
                    text = i.split(' ')
                    i = text.pop(-1)
                    listTo = ' '.join([str(elem) for elem in text]) 
                    strr += "\"" + listTo + "\" "
                strr += i + " "
            strr = strr[:-1]
            strr +=";"
        prediction_writer.writerow([key, strr])

with open(output_count, mode = 'w', newline='') as count:
    count_writer = csv.writer(count, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    count_writer.writerow(['ImagePath', 'PredictionString (class count;)'])
    for key in detections:
      c = detections[key][1]
      strr =''
      for tool in c:
        strr += "\"" + str(tool) + "\" " + str(c[tool]) + ";"
      count_writer.writerow([key,strr])

print("done")
# #testing output
# with open(output_prediction, newline='') as pred:
#     csv_reader = csv.reader(pred, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         else:
#             print(row)

