# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:45:20 2020

@author: kanum
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Ignoring  warnings
import warnings
warnings.filterwarnings("ignore")

Training_path = r'train/'
Testing_path = r'val/'


def create_dataset(path,img_size=160):
    Data = []
    x = []
    y = []
    Categories = os.listdir(path)#listing subfolders in train/test path
    
    for category in Categories:
        path_new = os.path.join(path,category)
        class_num = Categories.index(category)
        
        for img in tqdm(os.listdir(path_new)):#using tqdm we are iterating files in subfolders
            try:
                # using MTCNN we are cropping the part of face in the image
                img_array = cv2.imread(os.path.join(path_new,img))
                detector = MTCNN()
                results = detector.detect_faces(img_array)#detect faces returns 5 attributes of the image, in this case we need bounding boxes in it
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = img_array[y1:y2, x1:x2]#slicing the face pixels in the whole image
                face = cv2.resize(face,(img_size,img_size))#resizing image into required shape
                Data.append((face,class_num))#formig an Data array in hich it consists faces and there respective labels
                
            except Exception as e:
                pass
            
    for image,label in Data:
        x.append(image)
        y.append(label)
        
    return np.asarray(x), np.asarray(y)

#load Training dataset
x_train, y_train= create_dataset(Training_path)
print('Training data shape : ',x_train.shape, y_train.shape)
# load test dataset
x_test, y_test = create_dataset(Testing_path)  
print('Testing data shape : ',x_test.shape, y_test.shape) 

print("-------------------Data is Loaded-------------------")
#Load our model with predefined weights
model = load_model('facenet_keras.h5')
print('------------------Model is loaded-------------------')

#below Function returns embedded vector for a given image

def get_embed_vector(model,face):
    #Normalizing the data
    face = face/255.0
    mean,std = face.mean(),face.std()
    face=(face - mean)/std
    samples = np.expand_dims(face, axis=0)
    y_pred = model.predict(samples)
    
    return y_pred[0]

#convert every face_array into a embedding vector

def create_embed_vector(face_array):
    new_face_array=[]
    for face in face_array:
        embed = get_embed_vector(model,face)
        new_face_array.append(embed)
    return np.asarray(new_face_array)
        
#change x_train and x_test to arrays of embed vectors
       
new_x_train = create_embed_vector(x_train)
print("New training_data shape is : ",new_x_train.shape, y_train.shape)

new_x_test = create_embed_vector(x_test)
print("New test_data shape is : ",new_x_test.shape, y_test.shape)

#modulate the data to train a classifier
#scaling the embed vectors into length of 1
NORM = Normalizer(norm='l2')
new_x_train = NORM.transform(new_x_train)
new_x_test = NORM.transform(new_x_test)

#defining our mode(usimg a linear classifier in the below)
model = SVC(kernel='linear', probability=True)
#training
model.fit(new_x_train, y_train)
# predict
y_pred = model.predict(new_x_test)
# score
score = accuracy_score(y_test, y_pred)
# summarize
print('Accuracy on test set = %.2f' % (score*100))


    
    
                
                
        
        