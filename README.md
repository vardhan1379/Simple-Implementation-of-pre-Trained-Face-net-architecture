# Simple-Implementation-of-pre-Trained-Face-net-architecture

Face-Net is a start-of-art face recognition algorithm. It is a 22-layers deep neural network that directly trains its output to be a 128-dimensional embedding. The loss function used in this network is called triplet loss.

This Pre-Trained Face-net architecture is used in classifying the 5 Celebrity Faces Dataset.It includes photos of: Ben Affleck, Elton John, Jerry Seinfeld, Madonna, and Mindy Kaling. This dataset is already splitted into training and validation

We will use an MTCNN model for detecting the faces in the celebrity images,the FaceNet model will be used to create a face embedding(embedding vector) for each detected face, then we will develop a Linear Support Vector Machine (SVM) classifier model to classify the images.

Extract train.rar and val.rar in the same folder and download the facenet pre trained network from below link and add it to the same folder, click on here [face_net_keras.h5](https://www.kaggle.com/suicaokhoailang/facenet-keras)

