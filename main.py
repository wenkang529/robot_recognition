
# coding=utf-8
import cv2
import sys
import argparse
import face_recog.face_main
import barcode.barcode
from face_recog.face_main import Face
import digit.digit_main
from digit.recognize_digit.recognition_tensorflow import Digit_recognition
import threading ,Queue
import time
import numpy as np
import copy

quequ_draw=Queue.Queue(10)
main_img=[]

def thread_read(cap,thread_lock):
    # print('start read')
    global main_img
    while True:
        ret,data=cap.read()
        thread_lock.acquire()
        # print('read get lock')
        main_img=data
        thread_lock.release()
        # time.sleep(0.01)

def thread_show(thread_lock):
    print('start show')
    global main_img
    old_data = []
    while True:
        thread_lock.acquire()
        # print 'show get lock'
        img_data=main_img
        thread_lock.release()

        if len(img_data)>0:
            # print(np.shape(img_data))
            cv2.imshow('video',img_data)
            cv2.waitKey(1)
        if quequ_draw.qsize()>0:
            draw_wait=quequ_draw.get()

            if draw_wait.get('face') is not None:
                face_result = draw_wait.get('face')
                face_pic=draw_wait.get('pic')
                if face_result is not None:
                    print('face-result', len(face_result))
                    for face in face_result:
                        print(face.name)
                        bounding_box = face.bounding_box
                        cv2.rectangle(face_pic, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]),
                                      (255, 0, 0), 2)
                        print(face.name)
                        if face.name < 0.3 or True:
                            label = face.label
                            print('face-recognition:', label)
                            cv2.rectangle(face_pic, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]),
                                          (255, 0, 0), 2)
                            cv2.putText(face_pic, str(label), (bounding_box[0], bounding_box[3]), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), thickness=2, lineType=2)
                    cv2.imshow('face_find',face_pic)
            if draw_wait.get('digit_num') is not None:
                digit_num=draw_wait['digit_num']
                dashboard_position=draw_wait['dashboard_position']
                digits_position=draw_wait['digits_position']
                digit_pic=draw_wait['pic']
                if digit_num is not None:
                    print 'digit-num:', digit_num
                    cv2.rectangle(digit_pic, (dashboard_position[0], dashboard_position[1]),
                                  (dashboard_position[2], dashboard_position[3]), (100, 255, 0), 2)
                    for i in digits_position:
                        cv2.rectangle(digit_pic, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 2)
                    cv2.imshow('digit',digit_pic)
            if draw_wait.get('dash_x') is not None:
                dash_x=draw_wait['dash_x']
                dash_y=draw_wait['dash_y']
                dash_r=draw_wait['dash_r']
                dash_pic=draw_wait['dash_pic']
                if dash_x is not None:
                    cv2.circle(dash_pic, (dash_x, dash_y), dash_r, (0, 0, 255), 3)
                    cv2.circle(dash_pic, (dash_x, dash_y), 4, (0, 0, 255), -1)
                    cv2.imshow('dash_board',dash_pic)
            if draw_wait.get('barcode_number') is not None:
                barcode_pic=draw_wait['barcode_pic']
                barcode_point=draw_wait['barcode_point']
                barcode_comment=draw_wait['barcode_comment']
                barcode_number=draw_wait['barcode_number']
                if barcode_point is not None:
                    cv2.rectangle(barcode_pic, (barcode_point[0], barcode_point[1]), (barcode_point[2], barcode_point[3]),
                                  (0, 255, 0), 2)
                    cv2.putText(barcode_pic, str(barcode_number), (0, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
                    print(barcode_comment)
                    cv2.imshow('barcode',barcode_pic)

        # time.sleep(0.01)
        old_data=img_data

def thread_deal_face(thread_lock):
    # cv2.namedWindow('video')
    # cv2.namedWindow('face')
    global  main_img,quequ_draw
    print('start face')
    recognition = face_recog.face_main.Recognition(minsize=20)
    dict_face={}
    while True:
        thread_lock.acquire()
        print('face get thread')
        pic=main_img
        thread_lock.release()
        if len(pic)<=0:
            print('no pic for face ,sleep',len(pic))
            continue
        else:
            print(len(pic))


        face_result = recognition.recogniton(pic)
        dict_face['face']=face_result
        dict_face['pic']=pic
        quequ_draw.put(dict_face)

def thread_deal_digit(thread_lock):
    global  main_img,quequ_draw
    print('start digit')
    dict_digit={}
    while True:
        thread_lock.acquire()
        # print('digit get thread')
        pic=main_img
        thread_lock.release()
        if len(pic)<=0:
            # print('no pic for barcode ,sleep',len(pic))
            continue
        digit_num, dashboard_position, digits_position = digit.digit_main.digits_result(pic)
        dict_digit['digit_num']=digit_num
        dict_digit['dashboard_position']=dashboard_position
        dict_digit['digits_position']=digits_position
        dict_digit['pic']=pic
        quequ_draw.put(dict_digit)

def thread_deal_dashbord(thread_lock):
    global  main_img,quequ_draw
    print('start digit')
    dict_dashboard={}
    while True:
        thread_lock.acquire()
        # print('barcode get thread')
        dashboard_pic=main_img
        thread_lock.release()
        if len(dashboard_pic)<=0:
            # print('no pic for face ,sleep',len(dashboard_pic))
            continue
        dash_x,dash_y,dash_r=barcode.barcode.dash_bord(dashboard_pic)
        dict_dashboard['dash_x']=dash_x
        dict_dashboard['dash_y']=dash_y
        dict_dashboard['dash_r']=dash_r
        dict_dashboard['dash_pic']=dashboard_pic
        quequ_draw.put(dict_dashboard)

def thread_deal_barcode(thread_lock):
    global  main_img,quequ_draw
    print('start digit')
    dict_barcode={}
    while True:
        thread_lock.acquire()
        # print('barcode get thread')
        barcode_pic=main_img
        thread_lock.release()
        if len(barcode_pic)<=0:
            # print('no pic for face ,sleep',len(barcode_pic))
            continue
        barcode_point,barcode_comment,barcode_number=barcode.barcode.detect_barcode(barcode_pic)
        dict_barcode['barcode_point']=barcode_point
        dict_barcode['barcode_comment']=barcode_comment
        dict_barcode['barcode_number']=barcode_number
        dict_barcode['barcode_pic']=barcode_pic
        quequ_draw.put(dict_barcode)


if __name__=='__main__':
    thread_lock = threading.Lock()
    cap = cv2.VideoCapture(0)

    t_read = threading.Thread(target=thread_read,args=(cap,thread_lock))
    t_show = threading.Thread(target=thread_show,args=(thread_lock,))
    t_face = threading.Thread(target=thread_deal_face,args=(thread_lock,))
    t_digit = threading.Thread(target=thread_deal_digit,args=(thread_lock,))
    t_barcode = threading.Thread(target=thread_deal_barcode,args=(thread_lock,))
    t_dashboard = threading.Thread(target=thread_deal_dashbord,args=(thread_lock,))
    print('creat threads')

    t_read.start()
    t_show.start()
    t_face.start()
    t_digit.start()
    t_barcode.start()
    t_dashboard.start()








    # video_capture = cv2.VideoCapture(number)
    # digit_flag=0
    # # save_vedio=Save_video()
    # while video_capture.isOpened() :
    #     ret, frame_ori = video_capture.read()
    #     # face recognition
    #     face_result=recognition.recogniton(frame_ori)
    #     # barcode
    #     # barcode_point,barcode_comment,barcode_number=barcode.barcode.detect_barcode(frame_ori)
    #     barcode_point, barcode_comment,barcode_number=None,None,None
    #     #dashboard
    #     # dash_x,dash_y,dash_r=barcode.barcode.dash_bord(frame_ori)
    #     dash_x, dash_y, dash_r=None,None,None

