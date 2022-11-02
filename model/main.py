import numpy as np
import cv2
import pickle
import os

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_default.xml')

# load model
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

# load dictionary labels
labels = {}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    # đảo ngược key, value. vd person_name:1 chuyển thành 1:person_name
    labels = {v: k for k, v in og_labels.items()}

img_test = cv2.imread(os.getcwd()+'/test_data/1.jpg')
gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img_test[y:y+h, x:x+w]
    size = (128, 128)
    roi_color_resized = cv2.resize(roi_color, size)
    X_test = []
    X_test.append(roi_color_resized)
    X_test = np.array(X_test).reshape(1, -1)

    y_test = model.predict(X_test)  # dự đoán vs X_test

    # Thêm tên của người được dự đoán lên ảnh
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = labels[y_test[0]]
    color = (255, 255, 255)
    stroke = 2
    cv2.putText(img_test, name, (x, y), font,
                1, color, stroke, cv2.LINE_AA)

    # Vẽ vùng khuôn mặt
    color = (255, 0, 0)
    stroke = 2
    end_cord_x = x+w
    end_cord_y = y+h
    cv2.rectangle(img_test, (x, y), (end_cord_x, end_cord_y), color, stroke)

cv2.imshow('anh', img_test)
cv2.waitKey(5000)
cv2.destroyAllWindows()
