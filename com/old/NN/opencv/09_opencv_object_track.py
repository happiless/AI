import cv2
import numpy as np

img = cv2.imread('./data/book.jpg', cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)

sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_gray_frame, desc_gray_frame = sift.detectAndCompute(gray_frame, None)
    matches = flann.knnMatch(desc_image, desc_gray_frame, k=2)

    good_point = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_point.append(m)

    # img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)

    # Homography
    if len(good_point) > 10:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_point]).reshape(-1, 1, 2)
        trains_pts = np.float32([kp_gray_frame[m.trainIndx].pt for m in good_point]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, trains_pts, cv2.RANSAC, 0.5)
        matches_mask = mask.ravel().tolist()

        if matrix is None:
            continue

        h, w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        # 在每帧里面去画框的代码
        # [np.int32(dst)] : Array of polygonal curves
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

        cv2.imshow('homography', homography)
        
    else:
        cv2.imshow('gray_frame', gray_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
