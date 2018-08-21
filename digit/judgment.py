from digit.recognize_digit.recognition_ssd import seven_segment_display
from digit.process_image import process_image
from digit.recognize_digit.recognition_tensorflow import recognition_tensorflow


def judgment(recognition_tf, imgs):

    data, img_ssd = process_image(imgs)
    result = recognition_tensorflow(recognition_tf, data)

    for index, i in enumerate(result):
        if i == 1 or i == 3 or i == 4 or i == 5 or i == 8 or i == 9:
            try:
                number2 = seven_segment_display(img_ssd[index])
            except KeyError:
                continue
            if number2 == 0 or number2 == 6 or number2 == 7 or number2 == 9:
                result[index] = number2
    return result
