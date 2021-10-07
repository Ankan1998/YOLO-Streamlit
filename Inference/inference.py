from Inference.load_yolo_model import load_yolo_model
import cv2
import numpy as np
import time

class Inference_pipeline:

    def __init__(self,config_path):
        model, layer_output_name, labels, min_prob, nms_threshold = load_yolo_model(config_path)
        self.model = model
        self.layer_output_name = layer_output_name
        self.labels = labels
        self.min_prob = min_prob
        self.nms_threshold = nms_threshold


    def inference_image(self,img):

        img_blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

        self.model.setInput(img_blob)
        start = time.time()
        output = self.model.forward(self.layer_output_name)
        print("Time Taken ", time.time() - start)
        class_index = []
        bboxes = []
        conf = []
        h, w = img.shape[0], img.shape[1]

        for result in output:
            for object in result:
                score = object[5:]
                cls_idx = np.argmax(score)
                prob = score[cls_idx]
                if prob > self.min_prob:
                    bbox = object[:4] * np.array([w, h, w, h])
                    x_center, y_center, bbox_width, bbox_height = bbox
                    x_top = int(x_center - (bbox_width / 2))
                    y_top = int(y_center - (bbox_height / 2))

                    bboxes.append([x_top, y_top, int(bbox_width), int(bbox_height)])
                    conf.append(float(prob))
                    class_index.append(cls_idx)

        nms_result = cv2.dnn.NMSBoxes(bboxes, conf, self.min_prob, self.nms_threshold)

        return self.labels,nms_result, bboxes, conf, class_index

if __name__=="__main__":
    pass
