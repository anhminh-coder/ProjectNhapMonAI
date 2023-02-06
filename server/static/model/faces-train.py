from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import os
from PIL import Image
import pickle
from sklearn import decomposition
import pandas as pd
import scipy.stats as stats

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_default.xml')


id_ = 0
current_id = 0
label_ids = {}  # các cặp tên người ứng vs id. vd: person_name:1
X_train = []
y_train = []

# Detect khuôn mặt ra, tạo dữ liệu huấn luyện X_train, y_train
data_path = os.getcwd()+'/dataset'
for name in os.listdir(data_path):  # duyet tung thu muc trong dataset
    img_folder_path = data_path + '/' + name
    if name not in label_ids:
        label_ids[name] = current_id  # first_person:0
        current_id += 1  # tăng id người sau lên 1
    # duyet cac file anh trong tung thu muc
    for img in os.listdir(img_folder_path):
        img_path = img_folder_path + '/' + img
        img = cv2.imread(img_path)  # doc anh
        if img is None:
            continue
        # chuyen anh thanh mau xam
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for x, y, w, h in faces:
            roi = img[y:y+h, x:x+w]  # detect khuôn mặt
            size = (128, 128)
            # resize ảnh sau khi đã detect về cùng size
            roi_resized = cv2.resize(roi, size)
            roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
            X_train.append(np.array(roi_resized).flatten())  # thêm ảnh vào X_train
            y_train.append(np.array(id_))  # thêm label của ảnh vào y_train

    id_ += 1



X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train / 255

# pca = decomposition.PCA(n_components=100)
pca = decomposition.PCA(0.95)
pca.fit(X_train)
X_train = pca.transform(X_train)


with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)    

with open("pca.pickle", "wb") as f:
    pickle.dump(pca, f)

model = KNeighborsClassifier(n_neighbors=7, weights='distance')  # KNN
model.fit(X_train, y_train)

# lưu model
with open('model.pickle', "wb") as f:
    pickle.dump(model, f)