import streamlit as st
import cv2
import numpy as np
from PIL import Image
from Inference.inference import Inference_pipeline
from Inference.visualizer import Object_detection_visualizer
import tempfile
st.title("Object detection with YOLO")
config_path = r'C:\Users\Ankan\Desktop\Github\FastAPI-model-serving\Inference\config.json'
inf_pipeline = Inference_pipeline(config_path)
video_file_buffer = st.file_uploader("Upload a Video", type=["mp4"])
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(video_file_buffer.read())
video = cv2.VideoCapture(tfile.name)
stframe = st.empty()
if st.button("Detect Objects"):
    while True:
        ret, fr = video.read()
        labels, nms_result, bboxes, conf,class_idx = inf_pipeline.inference_image(fr)
        Object_detection_visualizer(fr, labels, nms_result, bboxes, conf, class_idx)
        img_ = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        stframe.image(img_, caption='Predicted Image.')