import pprint
import streamlit as st

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from edet_tflite import demo

def load_image():

    uploaded_file = st.file_uploader(
        label='Pick an image for object detection',
        type=['jpg', 'png', 'jpeg']
        )
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
    else:
        return None

def main():
    st.title('Demo for EfficientDet network for object detection with tensorflow lite')
    image_file = st.file_uploader(
        label='Pick an image for object detection',
        type=['jpg', 'png', 'jpeg']
    )
    if not image_file:
        return None
    else:
        st.image(image_file)
    n_boxes = st.slider("Number of objects to detect", min_value=1, max_value=20)

    result = st.button('Detect Objects')
    if result:
        img = cv2.cvtColor(np.array(Image.open(image_file)), cv2.COLOR_RGB2BGR)
        w = int(img.shape[0])
        h = int(img.shape[1])
        interpreter = demo.read_weights("weights/lite-model_efficientdet_lite0_detection_default_1.tflite")
        in_frame = demo.reshape_img(img)
        bboxes, class_ids, confs = demo.detect_objects(interpreter, in_frame)
        img = demo.draw_boxes(img, w, h, bboxes, n_boxes=n_boxes)
        img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_pil)
        st.image(img_pil)

if __name__ == '__main__':
    main()