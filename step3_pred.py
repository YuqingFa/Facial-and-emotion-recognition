import cv2
import os


modelFile = "./openCVmodel/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./openCVmodel/deploy.prototxt"
faceDetect = cv2.dnn_DetectionModel(modelFile, configFile)

facelabel = os.listdir("./data")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train.yml')


def face_detect_demo(image):
    detections = faceDetect.detect(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if len(detections) == 0:
        return None, None

    images = []
    for i in range(len(detections[2])):
        x, y, width, height = detections[2][i][0:4]
        img_split = cv2.resize(gray[y:y + width, x:x + height], (224, 224))
        images.append(img_split)

    return images, detections[2]


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)


def predict(image):
    img = image.copy()
    faces, rects = face_detect_demo(img)

    for index, face in enumerate(faces):
        rect = rects[index]
        label = recognizer.predict(face)

        if label[1] <= 50:
            label_text = facelabel[label[0]]
            draw_rectangle(img, rect)
            draw_text(img, label_text, rect[0], rect[1])

        else:
            draw_rectangle(img, rect)
            draw_text(img, "not find", rect[0], rect[1])

    return img


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        hasFrame, frame = cap.read()
        # 执行预测
        pred_img = predict(frame)
        cv2.imshow('detect', pred_img)
        k = cv2.waitKey(1)
        if ord('q') == cv2.waitKey(1):
            break
    cv2.destroyAllWindows()
    cap.release()
