# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM ultralytics/yolov5 

#Copy Contents and navigate to home
RUN mkdir /root/submission
COPY . /root/submission
WORKDIR /root
#COPY . /submission

#CMD ["python","detect.py", "--img 640", "--source test_imgs/", "--augment", "--conf 0.2", "--weights weights/custom/best.pt"]

