from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import os
from PIL import Image
import pickle
from sklearn import decomposition
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats


face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_default.xml')


current_id = 0
label_ids = {}  # các cặp tên người ứng vs id. vd: person_name:1
X_train = []
y_train = []
X_test = []
y_test = []
size = (128, 128)

# Tạo dữ liệu huấn luyện X_train, y_train
train_path = os.getcwd()+'/dataset/preprocessed/train'
for name in os.listdir(train_path):  # duyet tung thu muc trong dataset
    img_folder_path = train_path + '/' + name
    if name not in label_ids:
        label_ids[name] = current_id  # first_person:0
        current_id += 1  # tăng id người sau lên 1
    # duyet cac file anh trong tung thu muc
    for img in os.listdir(img_folder_path):
        img_path = img_folder_path + '/' + img
        img = cv2.imread(img_path) # doc anh
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        X_train.append(gray)
        y_train.append(current_id - 1)

X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = np.array(y_train)

# Tạo tập dữ liệu test X_test, y_test
current_id = 0
test_path = os.getcwd()+'/dataset/preprocessed/test'
for name in os.listdir(test_path):
    img_folder_path = test_path + '/' + name
    for img in os.listdir(img_folder_path):
        img_path = img_folder_path + '/' + img
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        X_test.append(gray)
        y_test.append(current_id)

    current_id += 1

X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], -1)
y_test = np.array(y_test)

# MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# PCA
pca = decomposition.PCA(0.95)
pca.fit(np.concatenate((X_train, X_test), axis=0))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# Tao dataframe
y_train = y_train.reshape((y_train.shape[0]), 1)
data = np.concatenate((X_train, y_train), axis=1)
df = pd.DataFrame(data=data)
df.rename(columns={df.columns[-1]: "label" }, inplace = True)

X_label_0 = df.loc[df.label == 0].iloc[:, :-1].values
X_label_1 = df.loc[df.label == 1].iloc[:, :-1].values
X_label_2 = df.loc[df.label == 2].iloc[:, :-1].values
X_label_3 = df.loc[df.label == 3].iloc[:, :-1].values


# Outlier detection
means = []
stds = []
def dectectOulier(X):
    outlier_indexes = []
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    means.append(mean)
    stds.append(std)
    z_score = (X - mean) / std
    for i in range(len(z_score)):
        for x in z_score[i]:
            if abs(x) >= 4:
                outlier_indexes.append(i)
                break
    
    cnt = 0
    for index in outlier_indexes:
        X = np.concatenate((X[:(index-cnt)],X[(index-cnt)+1 :]), axis=0)
        cnt += 1

    return X


X_label_0 = dectectOulier(X_label_0)
X_label_1 = dectectOulier(X_label_1)
X_label_2 = dectectOulier(X_label_2)
X_label_3 = dectectOulier(X_label_3)

X_train = np.concatenate((X_label_0, X_label_1, X_label_2, X_label_3), axis=0)
y_train = []
tmp = [X_label_0, X_label_1, X_label_2, X_label_3]
for i in range(len(tmp)):
    for _ in range(len(tmp[i])):
        y_train.append(i)

y_train = np.array(y_train)


mean_and_std = {}
mean_and_std["means"] = means
mean_and_std["stds"] = stds
 
# lưu dictionary labels bằng pickle
with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

with open("pca.pickle", "wb") as f:
    pickle.dump(pca, f)

with open("mean_and_std.pickle", "wb") as f:
    pickle.dump(mean_and_std, f)

model = KNeighborsClassifier(n_neighbors=5, weights='distance')    
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
acc = accuracy_score(y_true=y_test, y_pred= y_predicted)
print("Accuracy Score: ", acc)


# lưu model
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
