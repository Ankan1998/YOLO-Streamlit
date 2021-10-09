import cv2
import os
import json

def load_yolo_model(config_file):

    with open(config_file, 'r') as j:
        config = json.loads(j.read())

    try:
        with open(config['label_file']) as f:
            labels = [line.strip() for line in f]
        model = cv2.dnn.readNetFromDarknet(config['model_config'], config['yolo_weights'])

    except Exception as err:
        print(err)
        print("Your weights should be present here {}".format(os.getcwd()))
        exit(0)

    print("Model successfully loaded with weights")
    layers_name = model.getLayerNames()
    layer_output_name = [layers_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    return model, layer_output_name, labels, float(config['nms_threshold'])

if __name__=="__main__":
    config_file = r'C:\Users\Ankan\Desktop\Github\FastAPI-model-serving\inference\config.json'
    load_yolo_model(config_file)