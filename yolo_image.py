#python yolo_video.py --input car.mp4 --output car.avi
#python yolo_video.py --image
import sys
import argparse
#from yolo import YOLO, detect_video
from mobilenet import YOLO, detect_video
from PIL import Image
import numpy as np
import cv2

def detect_img(yolo):
            img = "test_data/london.jpg"
            image = Image.open(img)
            #print(image.width)
            r_image = yolo.detect_image(image)
            result = np.asarray(image)
            #r_image.show()
            #print(image.size)
            #print(result.shape)
            height, width, channels = result.shape
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result', width,height) 
            cv2.imshow("result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            yolo.close_session()


if __name__ == '__main__':
    
    detect_img( YOLO() )