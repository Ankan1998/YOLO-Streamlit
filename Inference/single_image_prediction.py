import cv2

from Inference.inference import Inference_pipeline
from Inference.visualizer import Object_detection_visualizer


def single_image_inference(img,config_path):
    inf_pipeline_obj = Inference_pipeline(config_path)
    labels, nms_result, bboxes, conf, class_idx = inf_pipeline_obj.inference_image(img)
    Object_detection_visualizer(img, labels, nms_result, bboxes, conf, class_idx)
    return img

if __name__ == "__main__":
    pass