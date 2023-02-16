import numpy as np
import os
import pathlib
import tensorflow as tf

from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2






# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/arseniy/Downloads/test/TensorFlow/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

     
os.chdir("/home/arseniy/main/or_wc/efficientdet_d2_coco17_tpu-32/saved_model/")
detection_model = tf.saved_model.load("/home/arseniy/main/or_wc/efficientdet_d2_coco17_tpu-32/saved_model")
     


detection_model.signatures['serving_default'].inputs

detection_model.signatures['serving_default'].output_dtypes

detection_model.signatures['serving_default'].output_shapes
     

def run_inference_for_single_image(model, image):
    image = cv2.resize(image, [768, 768], interpolation = cv2.INTER_AREA)
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
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
    # Visualization of the results of a detection.
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
    output_dict = run_inference_for_single_image(detection_model, frame)
    cv2.imshow('image window', cv2.resize(show_inference(frame, output_dict),[1600,720]))
    for i in range(30):
        ret, frame = cap.read()
        cv2.imshow('image window', cv2.resize(show_inference(frame, output_dict),[1600,720]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            t=1
            break
    if (cv2.waitKey(2) & 0xFF == ord('q')) or t:
        cv2.destroyAllWindows()
        break
    
