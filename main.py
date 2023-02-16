import numpy as np
import os

import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2

utils_ops.tf = tf.compat.v1

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

path_to_labels = '/home/arseniy/Downloads/test/TensorFlow/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

path_to_model="/home/arseniy/main/or_wc/efficientdet_d2_coco17_tpu-32/saved_model"
os.chdir(path_to_model)
detection_model = tf.saved_model.load(path_to_model).signatures['serving_default']

def run_inference_for_frame(model, image):

    input_tensor = tf.convert_to_tensor(image)

    input_tensor = input_tensor[tf.newaxis,...]

    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], 
            output_dict['detection_boxes'], 
            image.shape[0], 
            image.shape[1]
        )      
        detection_masks_reframed = tf.cast(
            detection_masks_reframed > 0.5, 
            tf.uint8
        )
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict
     

def show_inference(image_np, output_dict):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8
    )
    return image_np


cap = cv2.VideoCapture('rtsp://192.168.0.5:8080/h264_ulaw.sdp')
t=0
while(True):
    ret, frame = cap.read()
    output_dict = run_inference_for_frame(detection_model, frame)
    cv2.imshow('object_detection', cv2.resize(show_inference(frame, output_dict),[1600,720]))
    for i in range(40):
        ret, frame = cap.read()
        cv2.imshow('object_detection', cv2.resize(show_inference(frame, output_dict),[1600,720]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            t=1
            break
    if t:
        cv2.destroyAllWindows()
        break
    
