#corner detection
import numpy as np
import cv2

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(0,0), fx=2, fy=2)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#in corner detection, brg or rgb is almost always converted to grayscale as it is better for detecting edges or corners
    corners=cv2.goodFeaturesToTrack(gray,100,0.01,5) #args: source image or video,no. of top best corners, quality(0-1, 1 being perfect corner), minimum dist between corners returned
    corners=np.int0(corners) #takes np array(corners)and turns the float point values into integers
    for corner in corners:
        x,y=corner.ravel()
        cv2.circle(frame,(x,y),5,(0,0,255),-1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()