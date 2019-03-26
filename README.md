# Convert original keras Weight
## to weight (For training direct use)
    `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`

## to Model (For direct use)
    `python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5`

For VOC dataset, try `python voc_annotation.py`  

`python yolo_video.py --input test_data/akiha.mp4 --model trained_yolo.h5 --anchors anchors/yolo_anchors.txt --classes class/voc_classes.txt` 

`python yolo_video.py --input test_data/akiha.mp4 --model trained_yolo.h5 --anchors anchors/yolo_anchors.txt --classes class/voc_classes.txt` 

`python mobilenet_video.py --input test_data/akiha.mp4`

`python train.py --classes class_file test_data/akiha.mp4 -model model_file --anchors anchor_file`

`python evaluate.py -c eval_config.json`


tensorboard --logdir=logs/000