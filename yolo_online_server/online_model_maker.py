# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import sys
sys.path.append(".")
import numpy as np
from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework import graph_io
from yolo3.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.yolo3.utils import letterbox_image
import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

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

class YOLO(object):
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
        self.sess = K.get_session()
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
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

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

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        self.yolo_model.summary()
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
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
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

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
    def make_tf_server_model(self,output_name):
        net_model = self.yolo_model
        K.set_learning_phase(0)
        print('input is :', net_model.input.name)
        model_base64_placeholder = tf.placeholder(
            shape=(None,416,416,3),dtype=tf.float32,name="model_input_b64_images"
        )
        model_base64_input = tf.multiply(model_base64_placeholder,1)
        out_boxes, out_scores, out_classes = net_model(model_base64_input)
        sess = K.get_session()

        builder = tf.saved_model.builder.SavedModelBuilder(output_name)
        # x 为输入tensor, keep_prob为dropout的prob tensor 
        inputs = {'images': tf.saved_model.utils.build_tensor_info(model_base64_placeholder)}
        # y 为最终需要的输出结果tensor 
        outputs = {
                "out_boxes": tf.saved_model.utils.build_tensor_info(out_boxes),
                "out_scores": tf.saved_model.utils.build_tensor_info(out_scores),
                "out_classes": tf.saved_model.utils.build_tensor_info(out_classes)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')

        builder.add_meta_graph_and_variables(sess, ['serve'], {'test_signature':signature})
        builder.save()
        print('saved the constant graph (ready for inference) at: ', output_name)
    

    def close_session(self):
        self.sess.close()

output_name = 'yolo_online_server/modle/1'
output_fld = 'yolo_online_server/modle'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
yolo = YOLO()
yolo.make_tf_server_model(output_name=output_name)
