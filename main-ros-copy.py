# coding=utf-8

import cv2
import sys
import argparse
import face_recog.face_main
import barcode.barcode
from face_recog.face_main import Face
import digit.digit_main
import time
import rospy
import actionlib
from std_msgs.msg import String
from digit.recognize_digit.recognition_tensorflow import Digit_recognition

#############################

'''

tensorflow

cv2

scipy

sklearn

'''

#############################


def draw_pic(pic,face_result, barcode_point, barcode_comment, barcode_number,dash_x,dash_y,dash_r,digit_num, dashboard_position, digits_position):
    if face_result is not None:
        for face in face_result:
            # print(face.name)
            bounding_box = face.bounding_box
            cv2.rectangle(pic, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)
            if face.name < 0.3:
                label = face.label
                print('face-recognition:',label)
                cv2.rectangle(pic, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]),(255, 0, 0), 2)
                cv2.putText(pic, str(label), (bounding_box[0],bounding_box[3]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
    else:
        pass

    if barcode_point is not None:
        cv2.rectangle(pic, (barcode_point[0], barcode_point[1]), (barcode_point[2], barcode_point[3]),(0, 255, 0), 2)
        cv2.putText(pic, str(str(barcode_number)+'-->'+'message:'+barcode_comment), (0,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),thickness=2, lineType=2)
        print(barcode_comment)

    if dash_x is not None and 0:
        cv2.circle(pic, (dash_x, dash_y), dash_r, (0, 0, 255), -1)

    if digit_num is not None:
        print('digit-num:', digit_num)
        cv2.rectangle(pic, (dashboard_position[0], dashboard_position[1]), (dashboard_position[2], dashboard_position[3]), (255, 255, 0),2)
        for i in digits_position:
            cv2.rectangle(pic, (i[0], i[1]), (i[2], i[3]), (255, 255, 0),2)
    cv2.imshow('video', pic)


class Save_video:

    def __init__(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('out.avi', self.fourcc, 20.0, (320, 240))

    def save_video(self,pic):
        self.out.write(pic)

    def video_release(self):
        self.out.release()


def run_localcamera(number, minsize):
    recognition = face_recog.face_main.Recognition(minsize=minsize)
    video_capture = cv2.VideoCapture(number)
    
    global code_flag
    global qr_flag
    global dash_flag
    global digit_flag
    global face_flag

    while video_capture.isOpened():
        global work_done_Code
        global work_done_Face
        global work_done_Digit
        global work_done_QR
        global work_done_Instrument

        ret, frame_ori = video_capture.read()

        # face
        if work_done_Face == 'True' and face_flag:
            print("Face start!")
            face_result = recognition.recogniton(frame_ori)
            if face_result is not None:
                for face in face_result:
                    # print(face.name)
                    bounding_box = face.bounding_box
                    cv2.rectangle(frame_ori, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]),
                                  (255, 0, 0), 2)
                    if face.name < 0.3 or True:
                        label = face.label
                        print('face-recognition:', label)
                        cv2.rectangle(frame_ori, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]),
                                      (255, 0, 0), 2)
                        cv2.putText(frame_ori, str(label), (bounding_box[0], bounding_box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), thickness=2, lineType=2)
                    cv2.imshow('video', frame_ori)
                    cv2.waitKey(2000)

                if len(face_result) > 0:
                    work_done_Face = 'False'
                    face_flag=False
                    print("Face Done.")

        # barcode
        if work_done_Code == 'True' and code_flag:
            print("Barcode start!")
            barcode_point, barcode_comment, barcode_number=barcode.barcode.detect_barcode(frame_ori)
            if barcode_point is not None:
                cv2.rectangle(frame_ori, (barcode_point[0], barcode_point[1]), (barcode_point[2], barcode_point[3]),(0, 255, 0), 2)
                cv2.putText(frame_ori, str(barcode_number), (0,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),thickness=2, lineType=2)
                print(barcode_comment)
                cv2.imshow('video', frame_ori)
                cv2.waitKey(2000)
                work_done_Code = 'False'
                code_flag=False
                print("Barcode Done.")
                
        # QR code
        if work_done_QR == 'True' and qr_flag:
            print("QR code start!")
            barcode_point, barcode_comment, barcode_number=barcode.barcode.detect_barcode(frame_ori)
            if barcode_point is not None:
                cv2.rectangle(frame_ori, (barcode_point[0], barcode_point[1]), (barcode_point[2], barcode_point[3]),(0, 255, 0), 2)
                cv2.putText(frame_ori, str(barcode_number), (0,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),thickness=2, lineType=2)
                print(barcode_comment)
                cv2.imshow('video', frame_ori)
                cv2.waitKey(2000)
                work_done_QR = 'False'
                qr_flag=False
                print("QR code Done.")

        # digit
        if work_done_Digit == 'True' and digit_flag:
            print("Digit start!")
            digit_num, dashboard_position, digits_position = digit.digit_main.digits_result(frame_ori)
            if digit_num is not None:
                print('digit-num:', digit_num)
                cv2.rectangle(frame_ori, (dashboard_position[0], dashboard_position[1]),
                              (dashboard_position[2], dashboard_position[3]), (255, 255, 0), 2)
                for i in digits_position:
                    cv2.rectangle(frame_ori, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 2)
                cv2.imshow('video', frame_ori)
                cv2.waitKey(2000)
                work_done_Digit = 'False'
                digit_flag=False
                print("Digit Done.")

        # dashboard
        if work_done_Instrument == 'True' and dash_flag:
            print("Dashboard start!")
            dash_x,dash_y,dash_r=barcode.barcode.dash_bord(frame_ori)  
            if dash_x is not None and 0:
                cv2.circle(frame_ori, (dash_x, dash_y), dash_r, (0, 0, 255), 3)
                cv2.circle(frame_ori, (dash_x, dash_y), 4, (0, 0, 255), -1)
                cv2.imshow('video', frame_ori)
                cv2.waitKey(2000)
                work_done_Instrument = 'False'
                dash_flag=False
                print("Dashboard Done.")

        cv2.imshow('video', frame_ori)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def callback_Code(data):
    global work_done_Code
    global code_flag
    
    pub_Code = rospy.Publisher('Job_Code', String)
    rospy.loginfo("I heard %s", data.data)
    work_done_Code = 'True'
    while code_flag:
    	if work_done_Code == 'False':
    		break
    pub_Code.publish('Code_True')
    #print ("Done Code")


def callback_QR(data):
    global work_done_QR
    global qr_flag
    pub_QR = rospy.Publisher('Job_QR', String)
    rospy.loginfo("I heard %s", data.data)
    work_done_QR = 'True'
    while qr_flag:
    	if work_done_QR == 'False':
    		break
    pub_QR.publish('QR_True')

    #print ("Done QR")


def callback_Instrument(data):
    global work_done_Instrument
    global dash_flag

    pub_Instrument = rospy.Publisher('Job_Instrument', String)
    work_done_Instrument = 'True'
    while dash_flag:
    	if work_done_Instrument == 'False':
    		break
    pub_Instrument.publish('Instrument_True')

    #print ("Done Instrument")
def callback_Digit(data):
    global work_done_Digit
    global digit_flag
    pub_Digit = rospy.Publisher('Job_Digit', String)
    rospy.loginfo("I heard %s", data.data)
    work_done_Digit = 'True'
    while digit_flag:
    	if work_done_Digit == 'False':
    		break
    pub_Digit.publish('Digit_True')
    #print ("Done Code")

def callback_Face(data):
    global work_done_Face
    global face_flag
    pub_Face = rospy.Publisher('Job_Face', String)
    rospy.loginfo("I heard %s", data.data)
    work_done_Face = 'True'
    while face_flag:
        if work_done_Face == 'False':
            break
    pub_Face.publish('Face_True')
    print("Done Face")


def main(argv):

    minsize = argv.minsize
    local_index = argv.camera
    rospy.init_node('action')
    rospy.Subscriber("Location_Code", String, callback_Code)
    rospy.Subscriber("Location_QR", String, callback_QR)
    rospy.Subscriber("Location_Instrument", String, callback_Instrument)
    rospy.Subscriber("Location_Face", String, callback_Face)
    rospy.Subscriber("Location_Digit", String, callback_Digit)
    print("ros initilized")

    run_localcamera(local_index,minsize=minsize)

    rospy.spin()


def parse_arguments(argv):

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--url', default=None)

    parser.add_argument('--minsize', default=20 ,type=int)

    parser.add_argument('--camera',type=int,default=0)

    return parser.parse_args()


work_done_Code = 'False'
work_done_QR = 'False'
work_done_Instrument = 'False'
work_done_Face = 'False'
work_done_Digit = 'False'
code_flag=True
qr_flag=True
dash_flag=True
digit_flag=True
face_flag=True


if __name__=='__main__':

    main(parse_arguments(sys.argv[1:]))

