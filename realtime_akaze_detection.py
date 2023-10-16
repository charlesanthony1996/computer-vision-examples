import cv2
import numpy as np

def resize_image(image, scale_percent):
    width = int(image.shape[1]* scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize()
