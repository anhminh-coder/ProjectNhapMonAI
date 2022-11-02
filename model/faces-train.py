from cProfile import label
import numpy as np
import cv2
import os
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


id_ = 0
current_id = 0
label_ids = {}
X_train = []
y_train = []
data_path = os.getcwd()+'/dataset'
for name in os.listdir(data_path):
    img_folder_path = data_path + '/' + name
    if name not in label_ids:
        label_ids[name] = current_id
        current_id += 1
    for img in os.listdir(img_folder_path):
        img_path = img_folder_path + '/' + img
        pil_img = Image.open(img_path).convert('L')
        size = (550, 550)
        final_img = pil_img.resize(size, Image.Resampling.LANCZOS)
        img_array = np.array(final_img, 'uint8')

        faces = face_cascade.detectMultiScale(
            img_array, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in faces:
            roi = img_array[y:y+h, x:x+w]
            X_train.append(roi)
            y_train.append(id_)
    id_ += 1

with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(X_train, np.array(y_train))
recognizer.save('face-trainner.yml')
