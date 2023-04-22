#colors and color detection
#first convert bgr to hsv
import numpy as np 
import cv2

cap=cv2.VideoCapture(0)
while True:
    ret, frame= cap.read()
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #changing bgr to hsv
    lower_red=np.array([90,50,50]) #defining lower limit for color
    upper_red=np.array([130,255,255]) #defining upper limit for color
    mask=cv2.inRange(hsv, lower_red, upper_red) 
    x=cv2.bitwise_and(frame,frame,mask=mask) #bitwise function used for seperating blue from video
    cv2.imshow('frame', x)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()