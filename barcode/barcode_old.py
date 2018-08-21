from pyzbar.pyzbar import decode
import cv2
import numpy as np

def detect_barcode(pic):

    result=[]

    r1,point=None,None

    try:

        result = decode(pic)

    except:

        print('barcode err')

    if result != []:

        rect = result[0].rect

        point = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

        # print(point)

        # cv2.rectangle(pic, (point[0], point[1]), (point[2], point[3]),

        #               (0, 255, 0), 2)

        # cv2.putText(pic, str(result[0].data), (0,100),

        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),

        #             thickness=2, lineType=2)

        #print(str(result[0].data))

        r1=str(result[0].data)

    return point,r1

def dash_bord(frame):

    # res = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    res=frame

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    height, width = gray.shape[:2]

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, height / 10, 200, 100)


    if circles.size != 4:

        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles

        for (x, y, r) in circles:

            # draw the circle in the output image, then draw a rectangle

            # corresponding to the center of the circle

            if height * 0.1 < r < height * 0.4:

                return x,y,r

                # cv2.circle(frame, (x * 5, y * 5), r * 6, (0, 255, 0), 3)

                # cv2.circle(frame, (x * 5, y * 5), 4, (0, 0, 255), -1)

            else:

                return None,None,None

    else:

        return None,None,None