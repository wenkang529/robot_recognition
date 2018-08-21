from pyzbar.pyzbar import decode
import cv2
import numpy as np

def detect_barcode(pic):
    result = []

    r0, r1, point = None, None, None

    try:
        result = decode(pic)
    except:
        print('barcode err')

    if result != []:

        rect = result[0].rect

        point = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

        r0 = str(result[0].data)
        r0.encode('utf8')

        if r0 == '076149':
            r1='22335   105072  cn-aurora-07    Dell PowerEdge R610 Server  Standard    Dell    PowerEdge R610  1   $0.00   BJTT43X 76149   03/12/2015 0:00 Operational Active  "Beijing Lab (Beijing,S Wing of Tower C, Raycom InfoTech Park,8th)" APAC    China   Beijing "S Wing of Tower C, Raycom InfoTech Park"   8th Beijing Lab R06 Front   Internal    Front   24  No                              CN1079717   ESX Cloud Team-1001 gguanglu    gguanglu    CN1079628                       RnD_Ops_Central_Services    jkoppad     765138  02/06/2014      On'
        elif r0 == '076148':
            r1 = '22476   105073  cn-aurora-06    Dell PowerEdge R610 Server  Standard    Dell    PowerEdge R610  1   $0.00   CJTT43X 76148   03/12/2015 0:00 Operational Active  "Beijing Lab (Beijing,S Wing of Tower C, Raycom InfoTech Park,8th)" APAC    China   Beijing "S Wing of Tower C, Raycom InfoTech Park"   8th Beijing Lab R06 Front   Internal    Front   23  No                              CN1079717   ESX Cloud Team-1001 gguanglu    gguanglu    CN1079628                       RnD_Ops_Central_Services    jkoppad     765138  02/06/2014      On'
        elif r0 == '076144':
            r1 = '20254   105077  cn-aurora-02    Dell PowerEdge R510 Server  Standard    Dell    PowerEdge R510  2   $0.00   HCRCY2X 76144   03/12/2015 0:00 Operational Active  "Beijing Lab (Beijing,S Wing of Tower C, Raycom InfoTech Park,8th)" APAC    China   Beijing "S Wing of Tower C, Raycom InfoTech Park"   8th Beijing Lab R06 Front   Internal    Front   14  No                              CN1079717   ESX Cloud Team-1001 gguanglu    gguanglu    CN1079628                       RnD_Ops_Central_Services    jkoppad     563179  27/08/2013      On'
        elif r0 == '114422':
            r1 = '46315   231683  bjcusb031   HP ProLiant DL360 Gen9  Server  Standard    Hewlett-Packard DL360 Gen9  1   "$5,896.17 "    CN7651004K  114422  07/12/2017 0:00 Operational Active  "Beijing Lab (Beijing,S Wing of Tower C, Raycom InfoTech  Park,8th)"    APAC    China   Beijing "S Wing of Tower C, Raycom InfoTech Park"   8th Beijing Lab R01 Front   Internal    Front   40  No                              CN1079713   HaaS.-1001  HaaS    suns    CN1072400   CPBU            jnadar      RITM0111537     126946  On  RITM0111537     Received'
        else:
            r1 = r0

    return point, r1, r0



def dash_bord(frame):

    # res = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    res=frame

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    height, width = gray.shape[:2]

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, height / 5, 100, 200)


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
