import cv2
import tensorflow as tf

cap = cv2.VideoCapture('rtsp://192.168.0.5:8080/h264_ulaw.sdp')

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame,)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break