#python yolo_video.py --input car.mp4 --output car.avi
#python yolo_video.py --input test_data/akiha.mp4
#python yolo_video.py --image
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

#detect_video(YOLO(), 0 ,"saved_cam.avi")


detect_video(YOLO(), "http://192.168.43.232:8080/video" ,"saved_cam.mp4")