import cv2
import os

# 拍摄足够的数据
if __name__ == '__main__':
    name = "your_name"
    num = 200

    if not os.path.exists(os.path.join("data", name)):
        os.mkdir(os.path.join("data", name))

    cap = cv2.VideoCapture(0)
    for i in range(num):
        ret, frame = cap.read()
        cv2.imwrite("./data/{}/camera_{}.jpg".format(name, i), frame)

    cap.release()
