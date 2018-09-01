import cv2
from darkflow.net.build import TFNet
from utils.shape.rectangle import rectangle_properties
from capture_screen import grab_screen


def detect_objects(tfnet, imgcv):
    results = tfnet.return_predict(imgcv)
    colors = [(255, 0, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0)]
    vehicles = {'car', 'truck', 'bus', 'van'}
    beings = {'person'}

    for i, result in enumerate(results[:10]):
        rect = rectangle_properties(result)
        diagonal = rect.diagonal
        width = int(rect.width)
        length = int(rect.length)
        label = str(result["label"])
        box_label = label + ": {:.2f}".format(diagonal)
        color = colors[i % len(colors)]

        cv2.rectangle(imgcv,
                      (result["topleft"]["x"], result["topleft"]["y"]),
                      (result["bottomright"]["x"],
                       result["bottomright"]["y"]),
                      color, 2)

        text_x, text_y = result["topleft"][
                             "x"] - 10, result["topleft"]["y"] - 10
        if label in vehicles and diagonal >= 150:
            cv2.putText(imgcv, "Collision Warning: Vehicle", (300, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
        if label in beings and diagonal >= 110:
            cv2.putText(imgcv, "Collision Warning: Person", (300, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(imgcv, box_label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return imgcv


if __name__ == "__main__":
    options = {"model": "D:/darkflow/cfg/yolov2-tiny.cfg", "load": "D:/darkflow/bin/yolov2-tiny.weights",
               "threshold": 0.5,
               "labels": "D:/darkflow/labels.txt",
               "gpu": 0.3
               }

    tfnet = TFNet(options)

    while True:
        screen = grab_screen(region=(0, 40, 800, 640))
        object_predictions = detect_objects(tfnet, screen)
        # cv2.imshow('window', new_screen)
        # cv2.imshow('window2', cv2.cvtColor(object_predictions, cv2.COLOR_BGR2RGB))
        cv2.imshow('window2', cv2.cvtColor(object_predictions, cv2.COLOR_BGR2RGB))

        # if m1 < 0 and m2 < 0:
        #     right()
        # elif m1 > 0 and m2 > 0:
        #     left()
        # else:
        #     straight()

        # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
