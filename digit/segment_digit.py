from __future__ import division
import cv2
from imutils import contours
import imutils
import numpy as np
from digit.find_point import find_point



def segment_image(img_tran2seg, dashboard_position):

    # Resize image
    height, width = img_tran2seg.shape[:2]
    if width > height and width > 400:
        height = int(400 / width * height)
        width = 400
    elif width < height and height > 400:
        width = int(400 / height * width)
        height = 400
    size = height * width
    img_gray = cv2.resize(img_tran2seg, (width, height))

    # Calculate the average gray value of image; Image Processing
    sum_gray = np.sum(img_gray)
    avg_gray = int(sum_gray / size * 0.6)
    image_binarization = cv2.threshold(img_gray, avg_gray, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((2, 2), np.uint8)
    image_dilation = cv2.dilate(image_binarization, kernel, iterations=1)

    # Detect contour of digits
    (x0, y0, w0, h0) = dashboard_position
    digits_position = []
    cnts = cv2.findContours(image_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digit_cnts = []
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # if the contour is satisfied, it might be a digit
        if width*0.08 < w < width*0.4 and height*0.15 < h < height*0.4 and x < width*0.75 and w < h:
            x1 = x + x0
            y1 = y + y0
            digit_position = (x1, y1, x1+w, y1+h)
            digits_position.append(digit_position)
            digit_cnts.append(c)

    # find point and computing its relative position, if there is no digits then stop detecting point
    imgs = []
    point_position = 0
    if len(digit_cnts) != 0:

        i = 0
        array_x = []
        array_y = []
        array_w = []
        array_h = []
        range_h = []

        # sort the contours from left-to-right, then initialize the
        # actual digits themselves
        digitCnts = contours.sort_contours(digit_cnts, method="left-to-right")[0]
        for c in digitCnts:
            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            save_img = image_dilation[y:y + h, x:x + w]
            save_img = cv2.threshold(save_img, avg_gray, 255, cv2.THRESH_BINARY_INV)[1]
            imgs.append(save_img)
            i = i+1
            # extract the digit ROI
            array_x.append(x)
            array_y.append(y)
            array_w.append(w)
            array_h.append(h)

        if len(imgs) > 1:
            for j in range(i):
                h = array_y[j] + array_h[j]
                range_h.append(h)
            mid_img = img_gray[array_y[0]:max(range_h), array_x[0]:array_x[i - 1] + array_w[i - 1]]
            point_x, point_y, point_w, point_h = find_point(mid_img)
            if point_x != 0:
                for j in range(i):
                    if array_x[j] < point_x + array_x[0]:
                        point_position = j+1

    return imgs, point_position, digits_position




