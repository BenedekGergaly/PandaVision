from PIL import Image as pimg
import numpy as np
import cv2.aruco as aruco
import cv2
import time
import math
np.set_printoptions(precision=2, suppress=True)

camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

while True:
    _, img = camera.read()
    opencv_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (detected_corners, detected_ids, rejected_image_points) = aruco.detectMarkers(opencv_image_gray, aruco_dict)
    img_debug = aruco.drawDetectedMarkers(img, detected_corners, detected_ids, borderColor=(0, 255, 0))
    img_debug = aruco.drawDetectedMarkers(img_debug, rejected_image_points, borderColor=(0, 0, 255))
    img_small = cv2.resize(img_debug, (1536, 864))
    cv2.imshow("detected_corners", img_small)
    cv2.waitKey(1)