import cv2
import imutils
from auto_canny import auto_canny
import numpy as np



def transform_img(img_org):
    # gray_scale, blurring it, and computing an edge map
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edged = auto_canny(blurred)
    # find contours in the edge map, then sort them by their; size in descending order
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # find the biggest contour as the dashboard
    a_max = 0
    reg = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        a = w * h
        if a > a_max:
            a_max = a
            reg = c

    if len(reg) > 0:
        (x, y, w, h) = cv2.boundingRect(reg)
        img_tran2seg = img_gray[y:y + h, x:x + w]
        dashboard_position = [x, y, x+w, y+h]
    else:
        img_tran2seg = []
        dashboard_position = []

    return img_tran2seg, dashboard_position
