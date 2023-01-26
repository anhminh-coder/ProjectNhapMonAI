import numpy as np
import cv2
import pickle
import os
import sys
import pickle
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler

def dectectOulier(X, mean, std):
    z_score = (X - mean) / std
    for i in range(len(z_score)):
        for x in z_score[i]:
            if abs(x) >= 4:
                return True
    
    return False

def face_recognition(image_name):
    face_cascade = cv2.CascadeClassifier(
        'cascades/data/haarcascade_frontalface_default.xml')

    # load model
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    # Load mean and std
    with open('mean_and_std.pickle', 'rb') as f:
        mean_and_std = pickle.load(f)  

    # load dictionary labels
    labels = {}
    with open('labels.pickle', 'rb') as f:
        og_labels = pickle.load(f)
        # đảo ngược key, value. vd person_name:1 chuyển thành 1:person_name
        labels = {v: k for k, v in og_labels.items()}

    with open("pca.pickle", "rb") as f:
        pca = pickle.load(f)        

    img_test = cv2.imread(os.getcwd()+'/test_data/'+image_name)
    gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        size = (128, 128)
        roi_color_resized = cv2.resize(roi_gray, size)
        X_test = []
        X_test.append(roi_color_resized)
        X_test = np.array(X_test).reshape(1, -1)
        
        X_test = pca.transform(X_test)
        scaler = MinMaxScaler()
        X_test = scaler.fit_transform(X_test)
        means = mean_and_std['means']
        stds = mean_and_std['stds']
        checkOulier = False
        for i in range(len(means)):
            mean = means[i]
            std = stds[i]
            if dectectOulier(X_test, mean, std):
                checkOulier = True

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        stroke = 2
        if checkOulier == True:
            cv2.putText(img_test, "{}".format('undefined'), (x, y), font,
                    1, color, stroke, cv2.LINE_AA)

        else:
            y_test = model.predict_proba(X_test)[0]  # dự đoán vs X_test
            max_prob = -1
            idx = -1
            for i, prob in enumerate(y_test):
                print(y_test[i])
                if max_prob<y_test[i]:
                    max_prob = prob
                    idx = i
            # Thêm tên của người được dự đoán lên ảnh
            name = labels[idx]
            if max_prob > 0.5:
                cv2.putText(img_test, "{}: {:.2f}".format(name, max_prob), (x, y), font,
                        1, color, stroke, cv2.LINE_AA)
            
        # Vẽ vùng khuôn mặt
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(img_test, (x, y), (end_cord_x, end_cord_y), color, stroke)
        
    cv2.imwrite(os.getcwd()+'\\result_images\\'+image_name, img_test)
    cv2.imshow('anh', img_test)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if len(sys.argv) > 1:
    face_recognition(sys.argv[1])
else:
    print('es')
print('done')
print(os.getcwd())
face_recognition('4.jpg')