# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import requests
import json
from yolo3.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
class_name = {
'1':'停车痕紧',
'2':'停车痕松',
'3':'断经',
'4':'错花',
'5':'并纬',
'6':'缩纬',
'7':'缺纬',
'8':'糙纬',
'9':'折返',
'10':'断纬',
'11':'油污',
'12':'起机',
'13':'尽机',
'14':'经条',
'15':'擦白',
'16':'擦伤',
'17':'浆斑',
'18':'空织'
}

class YOLO_ONLINE(object):
    _defaults = {
        "model_path": 'yolo3/model_data/yolov3forflaw.h5',
        "anchors_path": 'yolo3/model_data/yolo_anchors.txt',
        "classes_path": 'yolo3/model_data/classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
    # _defaults = {
    #     "model_path": 'yolo3/model_data/yolov3.h5',
    #     "anchors_path": 'yolo3/model_data/yolo_anchors.txt',
    #     "classes_path": 'yolo3/model_data/coco_classes.txt',
    #     "score" : 0.3,
    #     "iou" : 0.45,
    #     "model_image_size" : (416, 416),
    #     "gpu_num" : 1,
    # }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self._t1 = 0.3
        self._t2 = 0.45
        #self.generate()
        self.sess = K.get_session()
        self.depth = (5+len(self.class_names))*3
        self.input1 = tf.placeholder(tf.float32,shape=(None,13,13,self.depth))
        self.input2 = tf.placeholder(tf.float32,shape=(None,26,26,self.depth))
        self.input3 = tf.placeholder(tf.float32,shape=(None,52,52,self.depth))
        self.boxes, self.scores, self.classes = self.generate()
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval([self.input1,self.input2,self.input3], self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes,scores, classes

    #"http://192.168.3.3:8500"
    def detect_image(self, image=None,endpoint = "http://127.0.0.1:8500"):
        start = timer()
        ans = {}
        ans['label']=[]
        ans['X']=[]
        ans['Y']=[]
        ans['W']=[]
        ans['H']=[]

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        print(image_data.shape)
        inputda = np.array(image_data).tolist()
        input_data = {
            "model_name": "default",
            "model_version": 1,
            "data": {
                "images": inputda}
        }
        result = requests.post(endpoint, json=input_data)
        #print(result.elapsed.total_seconds())
        end1 = timer()
        print(end1-start)
        new_dict = json.loads(result.text)
        out_boxes = np.array(new_dict['out_boxes'])
        out_scores = np.array(new_dict['out_scores'])
        out_classes = np.array(new_dict['out_classes'])
        output = [out_boxes,out_scores,out_classes]

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.input1: out_boxes,
                self.input2: out_scores,
                self.input3: out_classes,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        
        end2 = timer()
        print(end2-end1)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='yolo3/font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            ans['label'].append(str(predicted_class))
            ans['X'].append((left+right)/2)
            ans['Y'].append((top+bottom)/2)
            ans['W'].append(right-left)
            ans['H'].append(bottom-top)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image,ans


