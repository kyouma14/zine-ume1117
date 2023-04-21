import numpy as np
import cv2 

img1=cv2.imread(r'C:\Users\soham\Downloads\map3.PNG')
img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
i=0
for contour in contours:
    if i==0:
        i=1
        continue
    img1=cv2.drawContours(img1,contours,-1,(0,255,0),5)
    approx=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) #approximation function.
    M=cv2.moments(contour)
    x1=int(M["m10"]/M["m00"]) #centre of shape
    y1=int(M["m01"]/M["m00"])#centre of shape
    centrex=[x1]
    centrey=[y1]
    if len(approx)==3:
        cv2.putText(img1,"triangle",(x1-10,y1-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),5)
        cv2.circle(img1,(x1,y1),20,(255,255,0),-1)
    elif len(approx)==4:
            cv2.putText(img1,"square/rectangle",(x1-10,y1-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),5)
            cv2.circle(img1,(x1,y1),20,(255,255,0),-1)
    else:
        cv2.putText(img1,'Ellipse', (x1-10,y1-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),5)
        cv2.circle(img1,(x1,y1),20,(255,255,0),-1)
        img1=cv2.line(img1,(x1,y1),(x1,y1), (255,0,120),10) 
    cv2.imshow('image', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
