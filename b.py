#! /usr/bin/python2.7
import pickle

with open('new.pkl','rb') as f:
    data=pickle.load(f,protocol=3)
