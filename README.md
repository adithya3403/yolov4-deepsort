# yolov4-deepsort (ONLY IN COLAB)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b817NCQR8pZ3lQqkSdahz9GGywwN2rk5?usp=sharing)


## Clone repo and enable GPU
Enable GPU
```bash
!git clone https://github.com/adithya3403/yolov4-deepsort
%cd yolov4-deepsort
```

## Download weights

yolov4 weights: https://drive.google.com/file/d/10ks8KGAbsYG-ZXy2kBwd6TavnWAdLUtL/view?usp=share_link
tolov4-tiny weights: https://drive.google.com/file/d/1CKKmQ8y0uqKRUemJ6U0_QW7i7BSkpkuR/view?usp=share_link


## Running the Tracker with YOLOv4


```bash
# save yolov4-tiny model
!python save_model.py --weights /content/drive/MyDrive/yolov4-deepsort/yolov4.weights --model yolov4

# Run yolov4-tiny object tracker
!python object_tracker.py --video ./data/video/test3.mp4 --output ./outputs/demo33.mp4 --model yolov4 --dont_show
```

## Running the Tracker with YOLOv4-Tiny


```bash
# save yolov4-tiny model
!python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# Run yolov4-tiny object tracker
!python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test4.mp4 --output ./outputs/tiny4.mp4 --tiny --dont_show
```

## Resulting Video
The resulting video will save to wherever you set the ```--output``` command line flag path to.
I always set it to save to the ```outputs``` folder. 
