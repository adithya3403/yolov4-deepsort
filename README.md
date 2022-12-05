# yolov4-deepsort

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b817NCQR8pZ3lQqkSdahz9GGywwN2rk5?usp=sharing)


## Clone repo and enable GPU

For Colab:

Step1: Enable GPU

Step2: Clone repo

```bash
!git clone https://github.com/adithya3403/yolov4-deepsort
%cd yolov4-deepsort
```

## Running the Tracker with YOLOv4-Tiny

```bash
# save yolov4-tiny model
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# Run yolov4-tiny object tracker
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test4.mp4 --output ./outputs/tiny4.mp4 --tiny
```

## Resulting Video
The resulting video will save to wherever you set the ```--output``` command line flag path to.
Set it to save to the ```outputs``` folder. 
