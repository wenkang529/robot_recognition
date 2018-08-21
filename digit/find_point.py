import cv2
import imutils
import numpy as np


def find_point(img_gray):

    height, width = img_gray.shape[:2]
    size = height * width

    # Calculate the average gray value of image; Image Processing
    sum_gray = np.sum(img_gray)
    avg_gray = int(sum_gray / size)
    retval, binarization_image = cv2.threshold(img_gray, avg_gray, 255, cv2.THRESH_BINARY_INV)

    # Detect contour of digits
    cnts = cv2.findContours(binarization_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    point_Cnts = []
    digit_cnts = []
    point_x = 0
    point_y = 0
    point_w = 0
    point_h = 0

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # cv2.rectangle(img_gray, (x, y), (x + w, y + h), (0, 255, 0), 5)
        if 1 < w <= width*0.1 and 1 < h <= height*0.2 and y >= height*0.6 and x <= width*0.8:
            point_Cnts.append(c)
            # cv2.rectangle(img_gray, (x, y), (x + w, y + h), (0, 255, 0), 5)

            if len(point_Cnts) == 1:
                point_x = x
                point_y = y
                point_w = w
                point_h = h
        if w <= width * 0.5 and h >= min(height * 0.25, height * 0.5):
            digit_cnts.append(c)

    return point_x, point_y, point_w, point_h



