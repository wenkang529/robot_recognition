import cv2
import pickle
import numpy as np
# import face_recog.face_main
from face_recog.face_main import Face
import json
olddata='./face_recog/database/face_new_begin.pkl'
newdata='./face_recog/database/face_new_2.pkl'

# recog=face_recog.face_main.Recognition(minsize=20)
#
# with open(olddata,'rb')as f:
#     data=pickle.load(f)
#
# for i in data:
#     # i.face=cv2.cvtColor(i.face,cv2.COLOR_RGB2BGR)
#     if i.label=='wenkangw':
#         cv2.imshow('a',i.face)
#         cv2.waitKey(0)
# #
# with open(newdata,'r')as f:
#     datas=json.load(f)
#
# re=[]
#
# for i in datas:
#     face=Face()
#     face.face=i['face']
#     face.lalel=i['label']
#     re.append(face)
#    dic['face']=face
#    dic['label']=label
#    re.append(dic)


with open(olddata,'rb')as f:
    datas=pickle.load(f)


for i in datas:
    print(i.label)

# with open(newdata,'wb')as f:
#     pickle.dump(datas,f)


#############################for add pic to database
# recog=face_recog.face_main.Recognition(minsize=20)
# olddata = './face_recog/database/face_new_begin.pkl'
# newdata = './face_recog/database/face_new_begin2.pkl'
# pic=cv2.imread('./wenkangw.jpg')
# recog.database_add(olddata,newdata,pic,'wenkangw')
#####################################################

# cap=cv2.VideoCapture(0)
#
# while cap.isOpened() :
#     ret,fram=cap.read()
#     cv2.imshow('v',fram)
#     print('a')
#     if cv2.waitKey(1)==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()



