from __future__ import division
import numpy as np
from PIL import Image
import cv2


def process_image(imgs):
    data = []
    for img in imgs:
        height, width = img.shape[:2]
        if height > width:
            width = int((20 / height) * width)
            height = 20
        else:
            height = int((20 / width) * height)
            width = 20

        img_erosion = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(img_erosion, kernel, iterations=1)
        img_original = Image.fromarray(cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)).convert('L')
        img_bg = Image.open('./digit/recognize_digit/img_blank.jpg').convert('L')
        img_bg.paste(img_original, (int((28 - width) / 2), int((28 - height) / 2)))

        # data is used for recognition_tensorflow; since it needs array
        data1 = np.array(img_bg)
        for i in range(len(data1)):
            for j in range(len(data1[0])):
                data1[i, j] = 255 - data1[i, j]
        data1 = data1.reshape([784])
        data.append(data1)
    return data, imgs
