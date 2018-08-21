#! /usr/bin/python3
class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.embedding = None
        self.label=None
        self.face=None

import numpy as np
import pickle

a=[0]
b=a*128

data=np.array(b)

face=Face()
face.label='wenkangw'
face.embedding=data
r=[]
r.append(face)

with open('temp.pkl','wb') as f:
    pickle.dump(r,f)


