from PIL import Image
import numpy as np
import cv2

img = "test_data/london.jpg"

impil = Image.open(img)

print(impil.width)

imgcv = cv2.imread(img)

print(imgcv.width)