import cv2

img = cv2.imread('./data/white_panda.jpg')
img = cv2.GaussianBlur(img, (11, 11), 0)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1)

laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)

canny = cv2.Canny(img, 100, 150)

cv2.imshow('img', img)
cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)
cv2.imshow('sobel', sobel)
cv2.imshow('laplacian', laplacian)
cv2.imshow('canny', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
