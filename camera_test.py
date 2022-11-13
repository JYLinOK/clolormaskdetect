import cv2 
 
video_path = 1
# video_path = 'rtsp://url'

cap = cv2.VideoCapture(video_path)

while True:
    ref, frame = cap.read()
    cv2.namedWindow("curate and Fast Mask Recognition", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("curate and Fast Mask Recognition", 1920, 1080)
    cv2.imshow('curate and Fast Mask Recognition', frame)
    cv2.waitKey(1)