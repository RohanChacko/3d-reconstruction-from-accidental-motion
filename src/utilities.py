import cv2

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)