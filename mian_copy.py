# coding=utf-8
import cv2
import sys
import argparse
import face_recog.face_main
import barcode.barcode
from face_recog.face_main import Face
import digit.digit_main
from digit.recognize_digit.recognition_tensorflow import Digit_recognition
import time
#############################

'''
tensorflow
cv2
scipy
sklearn
'''
#############################


def draw_pic(pic,face_result,barcode_point,barcode_comment,barcode_number,dash_x,dash_y,dash_r,digit_num, dashboard_position, digits_position):
    if face_result is not None:
        for face in face_result:
            # print(face.name)
            bounding_box = face.bounding_box
            cv2.rectangle(pic, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)
            if face.name < 0.3:
                label = face.label
                print('face-recognition:',label)
                cv2.rectangle(pic, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]),(255, 0, 0), 2)
                cv2.putText(pic, str(label), (bounding_box[0],bounding_box[3]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),thickness=2, lineType=2)
    else:
        pass
    if barcode_point is not None:
        cv2.rectangle(pic, (barcode_point[0], barcode_point[1]), (barcode_point[2], barcode_point[3]),(0, 255, 0), 2)
        cv2.putText(pic, str(str(barcode_number)+'-->'+'message:'+barcode_comment), (0,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),thickness=2, lineType=2)
        print(barcode_comment)

    if dash_x is not None and 0:
        cv2.circle(pic, (dash_x, dash_y), dash_r, (0, 0, 255), -1)

    if digit_num is not None:
        print('digit-num:',digit_num)
        cv2.rectangle(pic, (dashboard_position[0], dashboard_position[1]), (dashboard_position[2], dashboard_position[3]), (255, 255, 0),2)
        for i in digits_position:
            cv2.rectangle(pic, (i[0], i[1]), (i[2], i[3]), (255, 255, 0),2)

    cv2.imshow('video',pic)

class Save_video:
    def __init__(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('out.avi', self.fourcc, 20.0, (320, 240))

    def save_video(self,pic):
        self.out.write(pic)

    def vedio_release(self):
        self.out.release()

def run_localcamera(number,minsize):
    recognition=face_recog.face_main.Recognition(minsize=minsize)
    video_capture = cv2.VideoCapture(number)
    digit_flag=0
    # save_vedio=Save_video()
    while video_capture.isOpened() :
        ret, frame_ori = video_capture.read()
        # face recognition
        face_result=recognition.recogniton(frame_ori)
        # barcode
        # barcode_point,barcode_comment,barcode_number=barcode.barcode.detect_barcode(frame_ori)
        barcode_point, barcode_comment,barcode_number=None,None,None
        #dashboard
        # dash_x,dash_y,dash_r=barcode.barcode.dash_bord(frame_ori)
        dash_x, dash_y, dash_r=None,None,None
        # dash board and digit
        if cv2.waitKey(1) == ord('d'):
            print('digit_press')
            digit_flag=1-digit_flag
            if digit_flag==1:
                print('digit_begin')
            else:
                print('digit_end')

        if digit_flag:
            digit_num, dashboard_position, digits_position = digit.digit_main.digits_result(frame_ori)
        else:
            digit_num, dashboard_position, digits_position=None,None,None
        # draw
        draw_pic(frame_ori,face_result,
                 barcode_point,barcode_comment,barcode_number,
                 dash_x,dash_y,dash_r,
                 digit_num, dashboard_position, digits_position)

        #save video
        # save_vedio.save_video(frame_ori)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    # save_vedio.vedio_release()


def main(argv):
    minsize=argv.minsize
    local_index=argv.camera
    run_localcamera(local_index,minsize=minsize)



def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--url', default=None)
    parser.add_argument('--minsize', default=20 ,type=int)
    parser.add_argument('--camera',type=int,default=0)
    return parser.parse_args()



if __name__=='__main__':

    main(parse_arguments(sys.argv[1:]))