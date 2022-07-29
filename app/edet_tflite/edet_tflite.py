import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image


def read_image(img_path):
    """
    This function reads image into cv2 format
    and returns img with its width and height
    """
    img = cv2.imread(img_path)
    w = int(img.shape[0])
    h = int(img.shape[1])

    return img, w, h


def decode_image_file(img_file):
    """
    Transform image from PIL format to cv2
    """
    cv2_img = cv2.cvtColor(np.array(Image.open(img_file)), cv2.COLOR_RGB2BGR)

    w = int(cv2_img.shape[0])
    h = int(cv2_img.shape[1])

    return cv2_img, w, h


def transform_cv_pil(cv2_img):
    """
    Transforms image from cv2 to PIL
    """
    img_pil = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_pil)

    return pil_img


def read_weights(weight_path):
    """
    This function reads weights and
    creates interpreter for inferencing of object detections
    """
    model_tflite = weight_path
    interpreter = tflite.Interpreter(model_path=model_tflite)
    interpreter.allocate_tensors()

    return interpreter


def reshape_img(img):
    """
    This function resizes and reshapes img
    preparing it for interpreter to analyze
    """
    in_frame = cv2.resize(img, (320, 320))
    in_frame = in_frame.reshape((1, 320, 320, 3))

    return in_frame


def detect_objects(interpreter, in_frame):
    """
    This function takes interpreter and image
    and returns boxes, classes, and confidences
    """

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, in_frame)
    interpreter.invoke()

    bboxes = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    class_ids = interpreter.get_tensor(interpreter.get_output_details()[1]["index"])
    confs = interpreter.get_tensor(interpreter.get_output_details()[2]["index"])

    return bboxes, class_ids, confs


def show_results_details(bboxes, class_ids, confs):
    """
    This function shows details of the model output
    """
    print("")
    print(bboxes.shape)
    print(bboxes)
    print(class_ids.shape)
    print(class_ids)  # We need to add +1 to the index of the result.
    print(confs.shape)
    print(confs)


def draw_boxes(img, w, h, bboxes, n_boxes=3):
    """
    This function draw boxes of objects on image
    """
    for n_box in range(n_boxes):
        box = bboxes[0][n_box]
        cv2.rectangle(
            img,
            (int(box[1] * h), int(box[0] * w)),
            (int(box[3] * h), int(box[2] * w)),
            (0, 255, 0),
            2,
            16,
        )
    return img


def write_labels(img, w, h, bboxes, n_boxes, class_ids):
    """
    This function write label's name of the object
    """

    for n_box in range(n_boxes):
        box = bboxes[0][n_box]
        cv2.putText(
            img,
            choose_label(int(class_ids[0][n_box])),
            (int(box[1] * h), int(box[2] * w)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            2,
        )

    return img


def show_image(img):
    """
    This function shows image in a window
    """
    cv2.imshow("demo", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def save_image(img, img_path):
    """
    This function saves image in the same folder
    where it was taken from with "_res" suffix
    """
    img_path = img_path.replace(".jpeg", "")
    cv2.imwrite(f"{img_path}_res.jpeg", img)  # saves demo result into results folder


def img_preprocess(
    img_path="images/dog.jpeg",
    weight_path="weights/lite-model_efficientdet_lite0_detection_default_1.tflite",
    n_boxes=3,
):
    """
    This function contains every preprocess step together
    and returns an statistical info, and image with boxes
    """
    img, w, h = read_image(img_path)
    interpreter = read_weights(weight_path)
    in_frame = reshape_img(img)
    bboxes, class_ids, confs = detect_objects(interpreter, in_frame)
    draw_boxes(img, w, h, bboxes, n_boxes)
    write_labels(img, w, h, bboxes, n_boxes, class_ids)

    return bboxes, class_ids, confs, img


def select_nboxes_conf(confs, conf_level):
    """
    Chooses number of boxes based on confidence level
    """
    nboxes = sum(confs.squeeze() >= conf_level)
    return nboxes


def choose_label(class_id):
    LABELS = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    return LABELS[class_id]
