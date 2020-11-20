import cv2
import time

cap = cv2.VideoCapture('./data/羊圈溜达模式.mp4')
_, first_frame = cap.read()

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
cv2.imshow('First Frame', first_frame)
cv2.imwrite('./data/liudayang.jpg', first_frame)

count_frame = 0
start = time.time()

while True:
    count_frame += 1
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    cv2.imshow('Frame', frame)
    cv2.imshow('Difference', difference)

    key = cv2.waitKey(30)
    if key == 27:
        break

    end = time.time()
    print("time %.2f s" % (end - start))
    print(count_frame)

cap.release()
cv2.destroyAllWindows()
