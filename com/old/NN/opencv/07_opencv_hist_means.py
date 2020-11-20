import cv2
import matplotlib.pyplot as plt

src = cv2.imread('./data/1.jpg', cv2.IMREAD_GRAYSCALE)
src = cv2.GaussianBlur(src, (11, 11), 0)

cv2.namedWindow('OriginalGrayImage')
cv2.imshow('OriginalGrayImage', src)

hist = cv2.calcHist([src], [0], None, [256], [0, 255])
plt.plot(hist, 'r')
plt.show()

plt.hist(src.ravel(), 256, [0, 256], color='r')
plt.show()

res = cv2.equalizeHist(src)
cv2.namedWindow('histogram equalization_opencv')
cv2.imshow('histogram equalization_opencv', res)

hist = cv2.calcHist([res], [0], None, [256], [0, 255])
plt.plot(hist, 'r')
plt.show()

plt.hist(res.ravel(), 256, [0, 256], color='r')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

