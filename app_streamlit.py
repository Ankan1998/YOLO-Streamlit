import streamlit as st
import cv2
import numpy as np
from PIL import Image
from Inference.single_image_prediction import single_image_inference
st.title("Object detection with YOLO")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
try:
    image_org = Image.open(img_file_buffer)
    st.image(image_org, caption='Uploaded Image.')
    img = cv2.cvtColor(np.array(image_org), cv2.COLOR_RGB2BGR)
    img_pred = single_image_inference(img,r'C:\Users\Ankan\Desktop\Github\FastAPI-model-serving\Inference\config.json')
    image_pred = Image.fromarray(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    st.image(image_pred, caption='Predicted Image.')
except:
    st.write("Please Upload an Image")

