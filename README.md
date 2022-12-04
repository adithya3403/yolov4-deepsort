# yolov4-deepsort (ONLY IN COLAB)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TDlx7in-rrTkCTwB1Y61l8T9Rc23jtkS?usp=sharing)

## Step 1:
Enable GPU
```bash
!git clone https://github.com/adithya3403/yolov4-deepsort
%cd yolov4-deepsort
```

## Step 2: Running the Tracker with YOLOv4-Tiny


```bash
# save yolov4-tiny model
python /content/yolov4-deepsort/save_model.py --weights /content/yolov4-deepsort/data/yolov4-tiny.weights --output /content/yolov4-deepsort/checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# Run yolov4-tiny object tracker
python /content/yolov4-deepsort/object_tracker.py --weights /content/yolov4-deepsort/checkpoints/yolov4-tiny-416 --model yolov4 --video /content/yolov4-deepsort/data/video/test.mp4 --output /content/yolov4-deepsort/outputs/tiny.avi --tiny
```

## Resulting Video
The resulting video will save to wherever you set the ```--output``` command line flag path to.
I always set it to save to the ```outputs``` folder. 
