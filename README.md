# zine-ume1117
Image Recognition

import numpy as np
import cv2

img1=cv2.imread(r'C:\Users\soham\Downloads\map3.PNG')
img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

img3=img1.copy()
img3=cv2.drawContours(img3,contours,-1,(0,255,0),5)
cv2.imshow('frame', img3)
cv2.imshow('f',thresh)
cv2.imshow('d',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
