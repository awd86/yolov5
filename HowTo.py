########################################
# Notes by/for Alex Denton
########################################

# The tutorial is here:
# https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/

# How to check for PyTorch installation and GPU compatability:
import torch
print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# To check the local Cuda version, use the terminal"
# nvcc --version

# To download the Udacity Self Driving Car Dataset in Yolov5 Pytorch Format:
# https://public.roboflow.com/ds/h0zYn5zFuK?key=tRsZIfO1Cg
# curl -L "https://public.roboflow.com/ds/h0zYn5zFuK?key=tRsZIfO1Cg" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

#################### Execution #########################

# Define model configuration and architecture (needed in runtime):
# define number of classes based on YAML
import yaml
with open(yolov5.data_car + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

# customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic


@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

# Train Custom YOLOv5 Detector - Next, we'll fire off training!
# Here, we are able to pass a number of arguments:
#
# img: define input image size
# batch: determine batch size
# epochs: define the number of training epochs. (Note: often, 3000+ are common here!)
# data: set the path to our yaml file
# cfg: specify our model configuration
# weights: specify a custom path to weights. (Note: you can download weights from the Ultralytics Google Drive folder)
# name: result names
# nosave: only save the final checkpoint
# cache: cache images for faster training

# train yolov5s on custom data for 100 epochs
# time its performance
# %%time
# %cd /content/yolov5/
#!python train.py --img 416 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache
# !python train.py --img [1920,1200] --batch 16 --epochs 100 --data {yolov5.data_car}/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache