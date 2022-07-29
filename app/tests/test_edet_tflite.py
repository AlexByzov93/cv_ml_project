import numpy as np
from edet_tflite import img_preprocess


def test_regr_img_preprocess():
    """Test if mean of bboxes is equal to previous mean on the same image"""

    bboxes, class_ids, confs, img = img_preprocess(img_path="images/dog.jpeg")
    assert np.isclose(bboxes.mean(), 0.31145704)
    assert np.isclose(np.linalg.norm(bboxes), 4.1600165)
    assert np.isclose(confs.mean(), 0.17875)


def test_img_preprocess_no_error():
    """Test if the preprocessing of the picture does not cause any error"""
    try:
        img_preprocess()
    except Exception:
        assert False, "img_preprocess raises exception for a regular size image"


def test_img_preprocess_no_error_sml():
    """Test if the preprocessing of the picture does not cause any error"""
    try:
        img_preprocess(img_path="images/dog_320x320.jpeg")
    except Exception:
        assert False, "img_preprocess raises exception for a small size image"
