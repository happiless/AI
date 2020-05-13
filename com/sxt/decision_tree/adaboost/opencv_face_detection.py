import cv2


def nothing(x):
    pass


# 开启摄像头
cap = cv2.VideoCapture(0)
# adaboost 识别人脸文件
face_cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
cv2.namedWindow('Frame')
cv2.createTrackbar('Neighbours', 'Frame', 2, 5, nothing)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    neighbours = cv2.getTrackbarPos('Neighbours', 'Frame')
    faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)
    for rect in faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
