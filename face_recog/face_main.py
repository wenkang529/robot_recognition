# coding=utf-8
import pickle
import os
import cv2
import numpy as np
import tensorflow as tf
import face_recog.align.detect_face
import face_recog.facenet

gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.dirname(__file__) + "//model_checkpoints//20170512-110547"

classifier_model = os.path.dirname(__file__) +'/database/a.pkl'
debug = False

olddata = './face_recog/database/new.pkl'
newdata = './face_recog/database/face_new_begin2.pkl'


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.embedding = None
        self.label=None
        self.face=None


class Detection:
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32,minsize=20):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.minsize=minsize
    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return face_recog.align.detect_face.create_mtcnn(sess, None)

    def face_predeal(self,crop):
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
        crop[:, :, 0] = cv2.equalizeHist(crop[:, :, 0])
        crop = cv2.cvtColor(crop, cv2.COLOR_YUV2BGR)
        return crop

    def pic_predeal(self,image, bounding_boxes, point_source):
        face_deal=[]
        for index, box in enumerate(bounding_boxes):
            box = box.astype(int)
            try:
                image_c = image[box[1]:box[3], box[0]:box[2], :]
                crop = cv2.resize(image_c, (self.face_crop_size, self.face_crop_size), interpolation=cv2.INTER_CUBIC)
            except:
                print('resize failure')
                print('img_shape:',np.shape(image))
                print('crop_shape:',np.shape(image_c))
                print('bounding_box',box)
                continue
            # histogram equalization
            crop=self.face_predeal(crop)
            # add to Face
            face=Face()
            face.face=crop
            face.bounding_box=box

            face_deal.append(face)
        return face_deal

    def find_faces(self, image):
        bounding_boxes,point_source = face_recog.align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        # nrof_faces = bounding_boxes.shape[0]  # 人脸数目
        # if nrof_faces >0:
        #     # print('找到人脸数目为：{}'.format(nrof_faces))
        # pic predeal
        return self.pic_predeal(image, bounding_boxes, point_source)


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            face_recog.facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = face_recog.facenet.prewhiten(face)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

class Identifier:
    def __init__(self):  #init in to init when build a new obj
        with open(classifier_model, 'rb') as f:
            self.data_set=pickle.load(f)

    #根据输入的人脸的embedding来输入神经网络，根据输出的最大概率的序号来确定哪个分类
    def identify(self, faces_iden):
        if faces_iden != []:
            for face in faces_iden:
                distance_list = []
                for j in self.data_set:
                    distance_list.append(np.sum(np.square(face.embedding - j.embedding)))
                distance_min = min(distance_list)
                # print(distance_list)
                index = distance_list.index(distance_min)
                face.label=self.data_set[index].label
                face.name=distance_min   # use name for return distance
        return faces_iden

class Recognition:
    def __init__(self,minsize):
        self.detect = Detection(minsize=minsize)
        self.encoder = Encoder()
        self.identifier = Identifier()

    def recogniton(self,pic):
        faces_detect=self.detect.find_faces(pic)
        if faces_detect == []:
            # print('no face return none')
            return None
        for i in faces_detect:
            i.embedding=self.encoder.generate_embedding(i.face)
        return self.identifier.identify(faces_detect)

    def database_change(self,olddata,newdata):
        with open(olddata, 'rb')as f:
            data = pickle.load(f)

        face_data = []
        for i in data:
            newface = Face()
            crop = i.face
            # histogram equalization
            crop=self.detect.face_predeal(crop)
            # add to Face
            newface.embedding = self.encoder.generate_embedding(crop)
            newface.face = i.face
            newface.label = i.label
            print(i.label)
            face_data.append(newface)
        with open(newdata, 'wb')as f:
            pickle.dump(face_data, f)

    def database_add(self,olddata,newdata,pic,label):
        with open(olddata, 'rb')as f:
            data = pickle.load(f)
        faces_detect=self.detect.find_faces(pic)
        if faces_detect == []:
            print('No face recognition')
            return None
        for i in faces_detect:
            i.embedding=self.encoder.generate_embedding(i.face)
            i.label=label
            data.append(i)
        with open(newdata,'wb')as f:
            pickle.dump(data,f)
