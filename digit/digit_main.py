import cv2
from digit.transform_img import transform_img
from digit.segment_digit import segment_image
from digit.judgment import judgment
from digit.recognize_digit.recognition_tensorflow import Digit_recognition

recognition_tf = Digit_recognition()

def digits_result(img_frame2digits):
    result_final = None
    try:
        img_tran2seg, dashboard_position = transform_img(img_frame2digits)
    except:
        return None, None, None
    digits_position = []
    if len(img_tran2seg) > 0:
        imgs, point_position, digits_position = segment_image(img_tran2seg, dashboard_position)
        if len(imgs) > 1:
            result = judgment(recognition_tf, imgs)
            result_len = len(result)
            if result_len > 0:
                result_final = 0
                if point_position != 0:
                    for i in range(result_len):
                        result_final += result[i] * 10 ** (point_position - i - 1)
                else:
                    for i in range(result_len):
                        result_final += result[i] * 10 ** (result_len - i - 1)
            else:
                dashboard_position = None
    return result_final, dashboard_position, digits_position


