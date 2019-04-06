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


tensorboard --logdir=logs/000 --port=6007

train batch size
yolo -> 32 , ?
small model -> 64 , 24

small mobilenet yolo for now
{0: 0.2886880500915588, 1: 0.49941118603127055, 2: 0.2444640965334233, 3: 0.16006168349920366, 4: 0.1948608902713586, 5: 0.5300083075861101, 6: 0.5409369836908366, 7: 0.5067628271579261, 8: 0.10056515957446807, 9: 0.3287429704172446, 10: 0.23961672473867596, 11: 0.41025285247998355, 12: 0.47350048872418604, 13: 0.4251144456578134, 14: 0.42984661736129237, 15: 0.09929483514389173, 16: 0.23318323608432168, 17: 0.3794570650635555, 18: 0.4837190072005488, 19: 0.3191247696198191}
aeroplane: 0.2887
bicycle: 0.4994
bird: 0.2445
boat: 0.1601
bottle: 0.1949
bus: 0.5300
car: 0.5409
cat: 0.5068
chair: 0.1006
cow: 0.3287
diningtable: 0.2396
dog: 0.4103
horse: 0.4735
motorbike: 0.4251
person: 0.4298
pottedplant: 0.0993
sheep: 0.2332
sofa: 0.3795
train: 0.4837
tvmonitor: 0.3191
mAP: 0.3444

small mobilenet 
first loss : ep003-loss113.347-val_loss94.183
middle loss : ep027-loss19.193-val_loss21.659.h5
last loss : last_loss14.0924-val_loss14.0924

Yolo
aeroplane: 0.7073
bicycle: 0.7195
bird: 0.5802
boat: 0.4040
bottle: 0.5104
bus: 0.7690
car: 0.7655
cat: 0.7586
chair: 0.4294
cow: 0.6622
diningtable: 0.5250
dog: 0.7139
horse: 0.7381
motorbike: 0.6654
person: 0.6838
pottedplant: 0.2941
sheep: 0.5743
sofa: 0.6178
train: 0.7244
tvmonitor: 0.5933
mAP: 0.6218