import cv2
import numpy as np
import collections
import math
from ObjectAndPosition import Object,Position


# 根据目标坐标位置计算判断目标是否进入或者离开
def checkTouchVSide(x, y, w, h, maxW, maxH, tolerance):
    if x <= 0:
        return True
    elif y - tolerance <= 0:
        return True
    elif x + w >= maxW:
        return True
    elif y + h + tolerance >= maxH:
        return True
    else:
        return False

#计算存储目标的颜色直方图
def getExteriorRect(pts):
    xArray = []
    yArray = []
    for pt in pts:
        xArray.append(pt[0])
        yArray.append(pt[1])
    xArray = sorted(xArray)
    yArray = sorted(yArray)
    return (xArray[0], yArray[0], xArray[3] - xArray[0], yArray[3] - yArray[0])

#计算检查目标在图像中的位置
def checkPosition(boundaryPt1, boundaryPt2, currPos, inCriterion):
    m = (boundaryPt2[1] - boundaryPt1[1])/(boundaryPt2[0] - boundaryPt1[0])
    c = boundaryPt2[1] - m*boundaryPt2[0]
    if inCriterion == "<":
        if currPos[0] * m + c < currPos[1]:
            return True
        else:
            return False
    elif inCriterion == ">":
        if currPos[0] * m + c > currPos[1]:
            return True
        else:
            return False
    else:
        return False

#CamShift跟踪器
def tracker(_x, _y, _w, _h, _frame):
    #获取目标坐标信息 
    x, y, w, h =  (int(_x), int(_y), int(_w), int(_h))
    frame = _frame
    track_window = (x,y,w,h)  #设置目标跟踪窗口
    roi = frame[y:y+h, x:x+w]  # 圈定感兴趣区域
    #颜色模型转换
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) # 创建感兴趣区域HSV色彩空间图像
    # inRange函数设置亮度阈值
    # 去除低亮度的像素点的影响
    # eg. mask = cv2.inRange(hsv, lower_red, upper_red)
    # 将低于和高于阈值的值设为0
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))  # 构建掩膜
    # 返回直方图
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180]) # 画未被遮盖部分的直方图(即只画mask中值为255的对应区域)
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX) # 线性归一化 norm_type=cv2.NORM_MINMAX 归一化到[α,β]之间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 创建当前帧的HSV色彩空间图像
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)  # 反向投影函数 找到感兴趣区域在原图中的位置
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)  # 迭代终止条件
                                            # EPS->达到精度停止迭代  COUNT->达到迭代次数停止迭代 EPS+COUNT->两者都作为条件
                                            # maxCount=10 最大迭代次数
                                            # epsilon=1 理想精度
    ret, track_window = cv2.CamShift(dst, track_window, term_crit) # 产生一个有方向的矩形追踪对象，并且大小可以随对象变化
    pts = cv2.boxPoints(ret) # 查找旋转矩形的四个顶点
    pts = np.int0(pts) # 将浮点数转换为整数
    img = cv2.polylines(frame,[pts],True, 255,2) # 画矩形
    return pts, img

if __name__=='__main__':
    cascade_src = 'myhaar.xml'
    video_src = 'cars.mp4'
    cap = cv2.VideoCapture(video_src)
    car_cascade = cv2.CascadeClassifier(cascade_src)
    bs = cv2.createBackgroundSubtractorKNN()    #创建背景减除器
    frames = 0  #记录帧数
    toleranceRange = 50  # 两帧间最大可允许目标移动距离
    toleranceCount = 10  # 最大可允许目标检测跟踪丢失帧数
    startHue = 0  # In HSV this is RED
    hueIncrementValue = 0.1  # 调整不同框线颜色
    midHeight = int(cap.get(4) / 2) #接下来是获取帧画面信息
    maxWidth = cap.get(3)  #Width of the frames in the video stream.
    maxHeight = cap.get(4)  #Height of the frames in the video stream.
    inCriterion = "<"
    boundaryPt1 = [0, midHeight-100]
    boundaryPt2 = [maxWidth, midHeight]
    detectedObj = [] #用来储存目标
    detectedContours = [] #储存新检测目标信息
    while True:
        ret, frame = cap.read()
        frames = frames + 1
        if (type(frame) == type(None)): #判断是否为最后一帧
            break
        myframe = frame.copy()
        fgmask = bs.apply(frame.copy())  # 前景掩码
        th = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)[1]  # 二值化
        eroded = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))  # 腐蚀
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)  # 膨胀
        # open=cv2.morphologyEx(th,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))) #开运算
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测轮廓
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 舍去过小的轮廓
                (x, y, w, h) = cv2.boundingRect(contour)
            else:
                continue
            #cv2.rectangle(myframe, (x, y), (x + w, y + h), (0, 255, 255), 1)
            roi = frame[y:y + h, x:x + w]  # 圈定感兴趣区域
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(roi_gray, 1.1, 3)  # 对感兴趣区域利用haar-cascade-classifier 检测
            for (x1,y1,w1,h1) in cars:
                detectedContours.append(Position(x+x1, y+y1, w1, h1))   #将坐标储存
                cv2.rectangle(myframe,(x+x1,y+y1),(x+x1+w1,y+y1+h1),(0,0,255),2)   #绘制检测框
        if len(detectedObj) != 0:
            for Obj in detectedObj:
                #获取目标坐标信息
                x, y, w, h =  (int(Obj.x), int(Obj.y), int(Obj.w), int(Obj.h))
                pts, img = tracker(x, y, w, h, myframe)
                pos = sum(pts)/len(pts)  #中心位置坐标
                isFound = False
                for dC in detectedContours: #判断该目标是否出现在新检测列表中
                    if (dC.x - toleranceRange < pos[0] < dC.x + dC.w + toleranceRange
                            and dC.y - toleranceRange < pos[1] < dC.y + dC.h + toleranceRange):
                        #如果是旧目标则更新其在确定物列表中的信息并将其从检测物列表中移除
                        Obj.setX(dC.x)
                        Obj.setY(dC.y)
                        Obj.setW(dC.w)
                        Obj.setH(dC.h)
                        Obj.setSpeed(pos - Obj.center)
                        Obj.setCenter(pos)
                        Obj.setMissingCount(0)
                        detectedContours.remove(dC)
                        isFound = True
                        tR = getExteriorRect(pts)
                        Obj.setRoi(frame[tR[1]:tR[1]+tR[3], tR[0]:tR[0]+tR[2]])
                        prevInStatus = Obj.isIn
                        currInStatus = checkPosition(boundaryPt1, boundaryPt2, Obj.center, inCriterion)
                        Obj.isIn = currInStatus
                #如果该目标不在新检测的目标中，则需判断目标是否丢失
                if not isFound:
                    if Obj.missingCount > toleranceCount:
                        #目标未检测跟踪到的帧数超过了阀值则判断为丢失，移出目标列表
                        detectedObj.remove(Obj)
                    else:
                        if checkTouchVSide(Obj.x + Obj.speed[0], Obj.y + Obj.speed[1], Obj.w,
                                           Obj.h, maxWidth, maxHeight, toleranceRange):
                        #如果目标离开画面，移出列表
                            detectedObj.remove(Obj)
                        else:
                        #如果未满足上述要求，则开始丢失帧数计数，且暂不移出列表
                            Obj.setMissingCount(Obj.missingCount+1)
                            Obj.setX(Obj.x + Obj.speed[0])
                            Obj.setY(Obj.y + Obj.speed[1])
                            Obj.setCenter(Obj.center + Obj.speed)

        #经过上述循环后，经过筛选的新检测目标列表中包含的信息均为新出现目标，写入到目标列表中
        for dC in detectedContours:
            if checkTouchVSide(dC.x, dC.y, dC.w, dC.h, maxWidth, maxHeight, toleranceRange):
                startHue += hueIncrementValue
                detectedObj.append(Object(dC.x, dC.y, dC.w, dC.h, frame[dC.y:dC.y+dC.h, dC.x:dC.x+dC.w], startHue))
                x, y, w, h =  (int(dC.x), int(dC.y), int(dC.w), int(dC.h))
                pts, img = tracker(x, y, w, h, myframe)

        # 清空新目标列表并显示画面
        detectedContours = []
        cv2.imshow('video', myframe)
        #cv2.imwrite(r'1result-'+str(frames)+'.jpg',frame)
        if cv2.waitKey(33) == 27:
            break
    cv2.destroyAllWindows()