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
st.subheader("Select Image or Video")
selected = st.radio("",["Images","Video"])
st.subheader("Confidence")
conf = st.slider("",
                 min_value=0.0,
                 max_value=1.0,
                 step=0.1,
                 value=0.5
                 )

if selected =="Video":
    video_file_buffer = st.file_uploader("Upload a Video", type=["mp4"])
    if video_file_buffer:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file_buffer.read())
        video = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        # if st.button("Detect Objects"):
        while True:
            ret, fr = video.read()
            labels, nms_result, bboxes, conf,class_idx = inf_pipeline.inference_image(fr,conf)
            Object_detection_visualizer(fr, labels, nms_result, bboxes, conf, class_idx)
            img_ = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            stframe.image(img_, caption='Object Detection on Video')
    else:
        st.error("Please Upload a video")
else:
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer:
        image_org = Image.open(img_file_buffer)
        img = cv2.cvtColor(np.array(image_org), cv2.COLOR_RGB2BGR)
        labels, nms_result, bboxes, conf, class_idx = inf_pipeline.inference_image(img,conf)
        Object_detection_visualizer(img, labels, nms_result, bboxes, conf, class_idx)
        img_ = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        st.image(img_, caption='Object Detection on Image')
    else:
        st.error("Please Upload a Image")