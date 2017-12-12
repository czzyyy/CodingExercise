# https://github.com/czzyyy/lbpcascade_animeface
# 用于检测出动漫的人脸
import cv2
import numpy as np
import os.path


def detect(filename, cascade_file='lbpcascade_animeface.xml'):
    cut_faces = []
    cut_faces_gray = []
    if not os.path.isfile(cascade_file):
        raise RuntimeError('%s not found ' % cascade_file)
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)  # read image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # change ro gray
    gray = cv2.equalizeHist(gray)  # 直方图均衡化

    # start detect faces
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(24, 24))

    for (x, y, w, h) in faces:
        # cut the face
        cut = image[y:y + h, x:x + w]
        cut_faces.append(cut)
        cut_gray = cv2.cvtColor(cut, cv2.COLOR_RGB2GRAY)  # change ro gray
        cut_faces_gray.append(cut_gray)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)
    cv2.imwrite(os.path.split(filename)[0] + '/out.png', image)

    return cut_faces, cut_faces_gray


if __name__ == '__main__':
    image_path = 'F:/python_code/girlface.jpg'
    cut_faces, cut_faces_gray = detect(image_path)
    for i in range(np.array(cut_faces).shape[0]):
        cv2.imwrite(os.path.split(image_path)[0] + '/' + str(i) + 'out.png', cut_faces[i])
