import cv2
import numpy as np

def Object_detection_visualizer(image,labels,nms_result,bboxes,conf,class_index):
    counter = 1
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    if len(nms_result) > 0:
        for i in nms_result.flatten():
            print('Object {}: {}'.format(counter, labels[int(class_index[i])]))
            counter += 1
            x_min, y_min = bboxes[i][0], bboxes[i][1]
            box_width, box_height = bboxes[i][2], bboxes[i][3]
            colour_box_current = colours[class_index[i]].tolist()
            cv2.rectangle(image, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            text_box_current = '{}: {:.4f}'.format(labels[int(class_index[i])],
                                                   conf[i])
            cv2.putText(image, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

