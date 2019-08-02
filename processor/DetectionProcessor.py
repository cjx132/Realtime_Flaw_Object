import sys
sys.path.append("..")
import cv2.cv2 as cv2
import numpy as np
#import datatime
from yolo3.yolo import YOLO
from PIL import Image, ImageFont, ImageDraw
class DetectionProcessor:
    def __init__(self):
        self.shape = (320,640)
        self.cnt = 0
    def object_detection(self,frames,yolo):
        ans={}
        ans1={}
        ans2={}
        if (isinstance(frames[0],np.ndarray)):        
            image = cv2.resize(frames[0],self.shape)
            image = Image.fromarray(image)
            image,ans = yolo.detect_image(image)
            self.frame01 = np.asarray(image)
        else:
            self.frame01 = None
        if (isinstance(frames[1],np.ndarray)):        
            #self.frame02 = cv2.resize(frames[1],self.shape)
            image1 = cv2.resize(frames[1],self.shape)
            image1 = Image.fromarray(image1)
            image1,ans1 = yolo.detect_image(image1)
            self.frame02 = np.asarray(image1)
        else:
            self.frame02 = None
        if (isinstance(frames[2],np.ndarray)):        
            #self.frame03 = cv2.resize(frames[2],self.shape)
            image2 = cv2.resize(frames[2],self.shape)
            image2 = Image.fromarray(image2)
            image2,ans2 = yolo.detect_image(image2)
            self.frame03 = np.asarray(image2)
        else:
            self.frame03 = None

        pack = {'frame01': self.frame01,'frame02': self.frame02,'frame03':self.frame03,'boxes1':ans,'boxes2':ans1,'boxes3':ans2}
        return pack

    
