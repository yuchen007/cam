import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time
from PIL import Image
import shutil
from collections import defaultdict

import io
# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("../..")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import label_map_util

################################################################################
import collections
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import PIL.Image as Image

txt_dict = {}
with open("id_convert.txt", 'r') as file_to_read:
  while True:
    lines = file_to_read.readline() 
    if not lines:
      break
      pass
    p_tmp, E_tmp = [float(i) for i in lines.split(":")] 
    txt_dict[int(p_tmp)] = int(E_tmp)
print("txt_dict:", txt_dict)
file_to_read.close()

class Predictor(object):
    """docstring for Predictor"""


    def __init__(self):
        self.NUM_CLASSES = 32
        self.PATH_TO_CKPT = './frozen_inference_graph.pb'  #模型名称
        self.PATH_TO_LABELS = './home_label_map.pbtxt'   #标签路径和名称
        self.detection_graph = self._load_gragh()
        self.category_index = self._load_mapphx()
        self.session = tf.Session(graph=self.detection_graph)
        self.im_width = 0
        self.im_height = 0


    def visualize_boxes_and_labels_on_image_array(self,
      image,
      boxes,
      classes,
      scores,
      category_index,
      instance_masks=None,
      instance_boundaries=None,
      keypoints=None,
      use_normalized_coordinates=False,
      max_boxes_to_draw=20,
      min_score_thresh=.7,
      agnostic_mode=False,
      line_thickness=4,
      groundtruth_box_visualization_color='black',
      skip_scores=False,
      skip_labels=False):
        ALLlist = []
        box_to_color_map = collections.defaultdict(str)


        if not max_boxes_to_draw:
          max_boxes_to_draw = boxes.shape[0]


        for box, color in box_to_color_map.items():
          ymin, xmin, ymax, xmax = box

          # print("box:", box)

        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
          if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            
            if scores is None:
              box_to_color_map[box] = groundtruth_box_visualization_color
            else:
              display_str = ''
              if not skip_labels:
                if not agnostic_mode:
                  if classes[i] in category_index.keys():
                      class_name = category_index[classes[i]]['name']
                      classes[i] = txt_dict[classes[i]]
                  else:
                      class_name = 'N/A'
                  display_str = str(class_name)

                  xmin = int(box[1] * self.im_width)
                  ymin = int(box[0] * self.im_height)
                  xmax = int(box[3] * self.im_width)
                  ymax = int(box[2] * self.im_height)
                  # x0 = int((xmin + xmax)/2)         #int(xmin + (xmax - xmin)/2)
                  # y0 = int((ymin + ymax)/2)         #int(ymin + (ymax - ymin)/2)
                  w = int(xmax - xmin)
                  h = int(ymax - ymin)
                  x0 = int(xmin + w/2)
                  y0 = int(ymin + h/2)
                  x_center = x0 + w/2
                  y_center = y0 + h/2
                  Tuple = (str(classes[i]), str(class_name), str(int(x_center)), str(int(y_center)), str(w), str(h), str(scores[i]))
                  Alist = list(Tuple)
                  ALLlist.append(Alist)
        return ALLlist


    def _load_gragh(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
              serialized_graph = fid.read()
              od_graph_def.ParseFromString(serialized_graph)
              tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_mapphx(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)

        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self,img):
        with self.detection_graph.as_default():
            # with self.session as sess:
                  # writer = tf.summary.FileWriter("logs/", sess.graph)
              self.session.run(tf.global_variables_initializer())
              
              self.im_height, self.im_width, c = img.shape
              image_np_expanded = np.expand_dims(img, axis=0)
              image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
              boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
              scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
              classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
              num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.

              (boxes, scores, classes, num_detections) = self.session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

        # *******************NMS************************************************#
              boxes = np.squeeze(boxes)

              classes = np.squeeze(classes).astype(np.int32)

              scores = np.squeeze(scores)

              # print("num_detections: ", num_detections)
              selected_indices=self.session.run(tf.image.non_max_suppression(boxes, scores, max_output_size=20, iou_threshold=0.4))
              #max_output_size: 一个整数张量，代表我最多可以利用NMS选中多少个边框
              # print("selected_indices: ", selected_indices)
              boxes=self.session.run(tf.gather(boxes,selected_indices))
              # print("selected_boxes: ", boxes)
              classes=self.session.run(tf.gather(classes,selected_indices))
              # print("classes:", classes[:5])
              scores=self.session.run(tf.gather(scores,selected_indices))
              # print("scores: ", scores[:5])
        # *******************NMS************************************************#

            # Visualization of the results of a detection.
              return self.visualize_boxes_and_labels_on_image_array(
                img,
                boxes,
                classes,
                scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=6,
                min_score_thresh=0.31)

            # if (average_time > 0.8):  #每张图片检测时间超过1s则重启程序
            #   os._exit(1)        
if __name__ == "__main__":
  #PATH_TO_TEST_IMAGES_DIR = 'D:/research/object_detection/transfered_dection_data/test/scene/ob/result/rgb_0301/'  #原始图片
  #image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, 'right-1573479058.jpg')
  image_path = './test.jpg'
  images = cv2.imread(image_path)
  prd = Predictor()
  for x in range(1,100):
    print(prd.detect(images))
  
  # print(predict(images))
  