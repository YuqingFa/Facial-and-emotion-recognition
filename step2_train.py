import os
import numpy as np
import cv2


# 脸部检测函数
def face_detect_demo(image):
    modelFile = "./openCVmodel/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "./openCVmodel/deploy.prototxt"
    faceDetect = cv2.dnn_DetectionModel(modelFile, configFile)

    detections = faceDetect.detect(image)
    if len(detections[0]) == 0:
        return None, None
    x, y, width, height = detections[2][0][0:4]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray[y:y + width, x:x + height], (224, 224)), detections[0]


def ReFileName(dirPath):
    faces = []
    for file in os.listdir(dirPath):
        if os.path.isfile(os.path.join(dirPath, file)) == True:
            c = os.path.basename(file)
            name = dirPath + '\\' + c
            img = cv2.imread(name)
            face, rect = face_detect_demo(img)
            if face is not None:
                faces.append(face)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces


if __name__ == '__main__':

    X, y = [], []
    for index, filename in enumerate(os.listdir("./data")):
        dirPath = r"./data/{}".format(filename)
        imgs = ReFileName(dirPath)

        for img in imgs:
            X.append(img)
            y.append(index)

    X, y = np.array(X).reshape((-1, 224, 224)), np.array(y).reshape(-1)

    index = [i for i in range(len(y))]
    np.random.seed(1)
    np.random.shuffle(index)
    train_data = X[index]
    train_label = y[index]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_data, train_label)
    recognizer.write('train.yml')
