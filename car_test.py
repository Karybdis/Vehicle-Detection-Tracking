import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('cars.mp4')  # open video
car_cascade=cv2.CascadeClassifier('myhaar.xml')
bs = cv2.createBackgroundSubtractorKNN()  # 背景减除
frames = 0  # 帧数
while True:
    ret, frame = cap.read()  # 读帧，ret是布尔值表示是否已经结束，frame是帧数据(三维)
    draw_frame=frame.copy()
    if type(frame) == type(None):
        break
    fgmask = bs.apply(frame.copy())  # 前景掩码
    if (frames < 5):
        frames += 1
        continue
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 对当前帧进行处理，创建灰度图像
    th = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)[1]  # 二值化图象th
    eroded = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))  # 腐蚀椭圆区域
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),iterations=2)  #eroded 膨胀椭圆区域,与上一条代码组合成开运算
    #open=cv2.morphologyEx(th,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))) #开运算
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测轮廓
    for contour in contours:  # contours是一个列表，表示轮廓信息
        if cv2.contourArea(contour) > 100:  # 计算轮廓面积
            (x, y, w, h) = cv2.boundingRect(contour)  # 得到外接矩形参数，xy是左上顶点坐标（不是笛卡尔坐标系），wh是宽高。https://www.cnblogs.com/gengyi/p/10317664.html
        else: continue
        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)  # 画矩形  参数1是原图，后两个参数分别是线条颜色宽度
        roi = frame[y:y + h, x:x + w]  # 圈定感兴趣区域
        roi_gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        cars=car_cascade.detectMultiScale(roi_gray,1.05,3)
        for (x1,y1,w1,h1) in cars:
            cv2.rectangle(draw_frame,(x1+x,y1+y),(x1+w1+x,y1+h1+y),(0,0,255),2)
    frames = frames + 1  # 帧数加一
    cv2.imshow('video', draw_frame)
    if cv2.waitKey(33) == 27:  # 返回x毫秒内按下键的编码，等于27(ESC)则退出  if 30frames/ms->33ms/frame
        break
cv2.destroyAllWindows()  # 已经循环完所有帧，关闭窗口
