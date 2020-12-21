import Augmentor
p= Augmentor.Pipeline(source_directory="/home/robot-data/Documents/Abhi/AI Hospital Challenge/yolov5/dataset_bing/images/validate", output_directory="/home/robot-data/Documents/Abhi/AI Hospital Challenge/yolov5/dataset_bing/images/aug_val", save_format="JPG")
p.random_brightness(probability=1, min_factor=0.5, max_factor=1.0)
#p.random_contrast(probability=1, min_factor=0.5, max_factor=1.0)
#p.greyscale(probability=1)
p.process()