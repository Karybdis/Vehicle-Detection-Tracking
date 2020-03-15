import colorsys

'''
构建一个类去储存检测到的目标的坐标位置以及尺寸大小
'''
class Position(object):
    def __init__(self, _x, _y, _w, _h):
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h

    def x(self):
        return self.x

    def y(self):
        return self.y

    def w(self):
        return self.w

    def h(self):
        return self.h


'''
目标类：包含确定目标的坐标位置，图像信息等信息
'''
class Object(object):
    def __init__(self, _x, _y, _w, _h, _roi, _hue):
        # 位置
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h
        self.roi = _roi

        # 控制框线颜色
        self.hue = _hue
        self.color = hsv2rgb(self.hue%1, 1, 1)

        # 运动位置信息
        self.center = [_x + _w/2, _y + _h/2]
        self.isIn = True
        self.isInChangeFrameCount = 0
        self.speed = [0,0]
        self.missingCount = 0

        # 颜色区域
        self.maxRoi = _roi
        self.roi = _roi

    def x(self):
        return self.x

    def y(self):
        return self.y

    def w(self):
        return self.w

    def h(self):
        return self.h

    def roi(self):
        return self.roi

    def color(self):
        return self.color

    def center(self):
        return self.center

    def maxRoi(self):
        return self.maxRoi

    def isIn(self):
        return self.isIn

    def speed(self):
        return self.speed

    def missingCount(self):
        return self.missingCount

    def isInChangeFrameCount(self):
        return self.isInChangeFrameCount

    def setX(self,value):
        self.x=value

    def setY(self,value):
        self.y=value

    def setW(self, value):
        self.w = value

    def setH(self, value):
        self.h = value

    def setCenter(self,value):
        self.center=value

    def setRoi(self,value):
        self.roi=value
        if self.roi.shape[0] * self.roi.shape[1] > self.maxRoi.shape[0] * self.maxRoi.shape[1]:
            self.maxRoi = self.roi

    def setSpeed(self,value):
        self.speed = value

    def setMissingCount(self,value):
        self.missingCount = value

    def setIsInChangeFrameCount(self,value):
        self.isInChangeFrameCount = value

def hsv2rgb(h, s, v):
    return tuple(i * 255 for i in colorsys.hsv_to_rgb(h, s, v))
