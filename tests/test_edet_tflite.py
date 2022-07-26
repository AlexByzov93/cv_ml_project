import numpy as np

from edet_tflite import demo


def test_regression_bboxes():
    """Test if mean of bboxes is equal to previous mean on the same image"""
    img_path = "images/dog.jpeg"
    weight_path = "weights/lite-model_efficientdet_lite0_detection_default_1.tflite"

    img, w, h = demo.read_image(img_path)
    interpreter = demo.read_weights(weight_path)
    in_frame = demo.reshape_img(img)
    bboxes, class_ids, confs = demo.detect_objects(interpreter, in_frame)
    assert np.isclose(bboxes.mean(), 0.33725205)


def test_img_preprocess_no_error():
    """Test if the preprocessing of the picture does not cause any error"""
    try:
        demo.img_preprocess()
    except Exception:
        assert False, "img_preprocess raises exception for a regular size image"


def test_img_preprocess_no_error_sml():
    """Test if the preprocessing of the picture does not cause any error"""
    try:
        demo.img_preprocess(img_path="images/dog_320x320.jpeg")
    except Exception:
        assert False, "img_preprocess raises exception for a small size image"
