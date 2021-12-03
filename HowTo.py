########################################
# Notes by/for Alex Denton
########################################

# The tutorial is here:
# https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/

# (to save requirements: pip freeze > requirements.txt)

# How to check for PyTorch installation and GPU compatability:

import torch
print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
#
# To check the local Cuda version, use the terminal"
# nvcc --version
#
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

import time
start = time.time()

# copy/paste this into the terminal
!python3 train.py --img 512 --batch 16 --epochs 200 --data ./data_car_small_sub/data.yaml --cfg ./models/customCAR_yolov5x.yaml --weights '' --name yolov5x_DGX4_results  --cache
# success on Udacity! Lambda

!python3 -m torch.distributed.run --nproc_per_node 4 train.py --img 512 --batch 64 --epochs 200 --data ./data_car_small_sub/data.yaml --cfg ./models/customCAR_yolov5x.yaml --weights '' --name yolov5x_DGX4_results  --cache
# success on Udacity! DGX4

#!python3 -m torch.distributed.run --nproc_per_node 4 train.py --img 512 --batch 64 --epochs 200 --data ./data_xView/xView.yaml --cfg ./models/xView_yolov5x.yaml --weights '' --name xView_yolov5x_results  --cache
# fail on xview 29 Nov: training unsuccessful - unstable, mAP <0.01

#!python3 -m torch.distributed.run --nproc_per_node 4 train.py --img 3200 --rect --batch 64 --epochs 200 --data ./data_xView/xView.yaml --cfg ./models/xView_yolov5x.yaml --weights '' --name xView_yolov5x_results  --cache
# fail on xivew 30 Nov: RuntimeError: stack expects each tensor to be equal size, but got [3, 2592, 3200] at entry 0 and [3, 3104, 3200] at entry 1

!python3 -m torch.distributed.run --nproc_per_node 4 train.py --img 1600 --batch 64 --epochs 200 --data ./data_xView/xView.yaml --cfg ./models/xView_yolov5x.yaml --weights '' --name xView_yolov5x_results  --cache
# WARNING: Extremely small objects found. 15256 of 537243 labels are < 3 pixels in size.
# fail on xView 30 Nov: RuntimeError: CUDA out of memory. Tried to allocate 98.00 MiB (GPU 3; 39.59 GiB total capacity; 36.46 GiB already allocated; 57.44 MiB free; 36.76 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF




# import train
# train(img=1920,batch=16,epochs=100,data='./data_car/data.yaml',cfg='./models/custom_yolov5s.yaml',name='yolov5s_results',cache=True)

# "--cache" can be " " (ram), "ram", or "disk" - but disk might really slow it down. It didn't seem to increase available cache size (21 nov 2021)

# "torch.distributed.run --nproc_per_node 4" per https://github.com/ultralytics/yolov5/issues/475
#  ** note: "--nproc_per_node 4" and "--batch 64" must match (choose '4' vise '5' for batch consideration)
# import torch
# use_cuda = torch.cuda.is_available()
#
# if use_cuda:
#     print('__CUDNN VERSION:', torch.backends.cudnn.version())
#     print('__Number CUDA Devices:', torch.cuda.device_count()) ### this one tells you how many are available
#     print('__CUDA Device Name:',torch.cuda.get_device_name(0))
#     print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

# "--rect" allows non-square images per: https://github.com/ultralytics/yolov5/issues/700

# the 'RuntimeError: non-positive stride is not supported' is here: https://github.com/ultralytics/yolov5/issues/1671
# the 'Invalid Syntax...async' was here: https://stackoverflow.com/questions/60842431/python3-8-use-async-getting-invalid-sysntax

# there was a major issue with the "custom_yolov5s.yaml" file - I don't remember how it was created
# the data needs to be in 3 folders: train, test, valid
# RAM overload: " resource_tracker: There appear to be 6 leaked semaphore objects to clean up at shutdown"