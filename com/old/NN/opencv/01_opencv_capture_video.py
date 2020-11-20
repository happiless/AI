import cv2

# 创建显示视频的窗口
cv2.namedWindow('Video')

# 打开摄像头
video_capture = cv2.VideoCapture(0)

# 创建视频写入对象
video_writer = cv2.VideoWriter('./data/test_img.mp4',
                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               video_capture.get(cv2.CAP_PROP_FPS),
                               (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                )
                               )

success, frame = video_capture.read()
while success and not cv2.waitKey(1) == 27:
    blur_frame = cv2.GaussianBlur(frame, (3, 3), 0)
    video_writer.write(blur_frame)
    cv2.imshow('Video', blur_frame)
    success, frame = video_capture.read()

cv2.destroyAllWindows()
video_capture.release()
