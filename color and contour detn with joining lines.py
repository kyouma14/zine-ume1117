import numpy as np 
import cv2 

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(0,0),fx=2,fy=2)

    

    #converting bgr to hsv
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #creating masks for different colors

#red
    low_red=np.array([136,87,111],np.uint8)
    high_red=np.array([180,255,255],np.uint8)
    mask_red=cv2.inRange(hsv,low_red,high_red)

    #green
    low_green=np.array([50,100,10],np.uint8)
    high_green=np.array([70,255,255],np.uint8)
    mask_green=cv2.inRange(hsv,low_green,high_green)
    
    #blue
    low_blue=np.array([94,80,2],np.uint8)
    high_blue=np.array([120,255,255],np.uint8)
    mask_blue=cv2.inRange(hsv,low_blue,high_blue)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    
    kernel=np.ones((5,5),"uint8")  #5x5 matrix consisting only of ones

    #for red
    mask_red=cv2.dilate(mask_red,kernel)  #kernal is sliding over the image, broadening red color
    res_red=cv2.bitwise_and(frame,frame,mask=mask_red)

    #for green
    mask_green=cv2.dilate(mask_green,kernel)
    res_green=cv2.bitwise_and(frame,frame,mask=mask_green)

    #for blue
    mask_blue=cv2.dilate(mask_blue,kernel)
    res_blue=cv2.bitwise_and(frame,frame,mask=mask_blue)

    #contours in red
    contours,heirarchy=cv2.findContours(mask_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt,contour in enumerate(contours):
        area=cv2.contourArea(contour)
        M=cv2.moments(contour)
        if M['m00']!=0:
            x1=int(M['m10']/M['m00'])
            y1=int(M['m01']/M['m00'])
        approx=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        if (area>300):
            x,y,w,h=cv2.boundingRect(contour)
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,"Red",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            if len(approx)==3:
                cv2.putText(frame,"Triangle",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            elif len(approx)==4:
                cv2.putText(frame,"Square/Rectangle",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            else:
                cv2.putText(frame,"Circle",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    
    #contours in green
    contours,heirarchy=cv2.findContours(mask_green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt,contour in enumerate(contours):
        M=cv2.moments(contour)
        if M['m00']!=0:
            x2=int(M['m10']/M['m00'])
            y2=int(M['m01']/M['m00'])
        approx=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area=cv2.contourArea(contour)
        if area>300:
            x,y,w,h=cv2.boundingRect(contour)
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,"Green",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            if len(approx)==3:
                cv2.putText(frame,"Triangle",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            elif len(approx)==4:
                cv2.putText(frame,"Square/Rect",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            else:
                cv2.putText(frame,"Circle",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))

    #contours in blue
    contours,heirarchy=cv2.findContours(mask_blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt, contour in enumerate(contours):
        M=cv2.moments(contour)
        if M['m00']!=0:
            x3=int(M['m10']/M['m00'])
            y3=int(M['m01']/M['m00'])
        approx=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area=cv2.contourArea(contour)
        if area>300:
            x,y,w,h=cv2.boundingRect(contour)
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,"Blue",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            if len(approx)==3:
                cv2.putText(frame,"Triangle",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            elif len(approx)==4:
                cv2.putText(frame,"Square/Rect",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            else:
                cv2.putText(frame,"Circle",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    frame=cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),5)
    frame=cv2.line(frame,(x2,y2),(x3,y3),(0,255,0),5)
    frame=cv2.line(frame,(x3,y3),(x1,y1),(255,0,0),5)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
        




