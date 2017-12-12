# coding:utf-8
import cv2
import os
import numpy as np


def detect_faces(filename):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cut_faces = []
    cut_faces_gray = []
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )
    for (x, y, w, h) in faces:
        # cut the face
        cut = image[y:y + h, x:x + w]
        cut_faces.append(cut)
        cut_gray = cv2.cvtColor(cut, cv2.COLOR_RGB2GRAY)  # change ro gray
        cut_faces_gray.append(cut_gray)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imshow("Find Faces!", image)
    cv2.waitKey(0)
    return cut_faces, cut_faces_gray

if __name__ == '__main__':
    filename = 'F:/python_code/girl.jpg'
    cut_faces, cut_faces_gray = detect_faces(filename)
    print(np.array(cut_faces).shape)
    for i in range(np.array(cut_faces).shape[0]):
        cv2.imwrite(os.path.split(filename)[0] + '/' + str(i) + 'face_out.png', cut_faces[i])
