import cv2
import numpy as np


def seven_segment_display(img):
    digit_model = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (0, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    height, width = img.shape[:2]
    c = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thresh[y:y + h, x:x + w]
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dhc = int(roiH * 0.05)
    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, 0), (int(roiW * 0.35), h // 2)),  # top-left
        ((w - dW, 0), (w, h // 2)),  # top-right
        ((0, (h // 2) - dhc), (w, (h // 2) + dhc)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((w - int(roiW * 0.35), h // 2), (w, h)),  # bottom-right
        ((0, h - dH), (w, h))  # bottom
    ]
    on = [0] * len(segments)

    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        seg_roi = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(seg_roi)
        area = (xB - xA) * (yB - yA)
        # print('%d rate:%f' % (i, total / float(area)))
        if area != 0:
            if float(total) / float(area) > 0.3:
                on[i] = 1

    digit = digit_model[tuple(on)]
    return digit
