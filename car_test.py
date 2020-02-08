import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('cars.mp4')
car_cascade=cv2.CascadeClassifier('myhaar.xml')
bs = cv2.createBackgroundSubtractorKNN()
frames = 0  
while True:
    ret, frame = cap.read()
    draw_frame=frame.copy()
    if type(frame) == type(None):
        break
    fgmask = bs.apply(frame.copy())  # 前景掩码
    if (frames < 5):
        frames += 1
        continue
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)[1]  # 二值化
    eroded = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))  # 腐蚀
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),iterations=2)  #膨胀
    #open=cv2.morphologyEx(th,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))) #开运算
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测轮廓
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 舍去过小的轮廓
            (x, y, w, h) = cv2.boundingRect(contour)
        else: continue
        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        roi = frame[y:y + h, x:x + w]  # 圈定感兴趣区域
        roi_gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        cars=car_cascade.detectMultiScale(roi_gray,1.05,3)  #对感兴趣区域利用haar-cascade-classifier 检测
        for (x1,y1,w1,h1) in cars:
            cv2.rectangle(draw_frame,(x1+x,y1+y),(x1+w1+x,y1+h1+y),(0,0,255),2)
    frames = frames + 1
    cv2.imshow('video', draw_frame)
    if cv2.waitKey(33) == 27:
        break
cv2.destroyAllWindows()
