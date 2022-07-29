import streamlit as st
from edet_tflite import decode_image_file
from edet_tflite import detect_objects
from edet_tflite import draw_boxes
from edet_tflite import read_weights
from edet_tflite import reshape_img
from edet_tflite import select_nboxes_conf
from edet_tflite import transform_cv_pil
from edet_tflite import write_labels


@st.cache
def read_weights_cached(
    weight_path="weights/lite-model_efficientdet_lite0_detection_default_1.tflite",
):
    return read_weights(weight_path)


@st.cache
def detect_objects_cached(interpreter, in_frame):
    return detect_objects(interpreter, in_frame)


def main():
    st.title("Demo for EfficientDet network for object detection with tensorflow lite")
    with st.sidebar:
        image_file = st.file_uploader(
            label="Pick an image for object detection", type=["jpg", "png", "jpeg"]
        )
        options = st.selectbox(
            "How would you like to choose number of objects?",
            ("Specify Confidence", "Specify Number of Objects"),
        )
        if options == "Specify Number of Objects":
            n_boxes = st.slider(
                "Number of objects to detect", min_value=1, max_value=20
            )
        else:
            conf_int = st.slider(
                "Confidence level", min_value=0.0, max_value=1.0, step=0.01
            )

        result = st.button("Detect Objects")

    if not image_file:
        return None
    else:
        st.image(image_file)

    if result:
        img, w, h = decode_image_file(image_file)
        interpreter = read_weights_cached()
        in_frame = reshape_img(img)
        bboxes, class_ids, confs = detect_objects_cached(interpreter, in_frame)

        n_box = (
            n_boxes
            if options == "Specify Number of Objects"
            else select_nboxes_conf(confs, conf_int)
        )

        img = draw_boxes(img, w, h, bboxes, n_boxes=n_box)
        img = write_labels(img, w, h, bboxes, n_box, class_ids)

        img_pil = transform_cv_pil(img)
        st.image(img_pil)


if __name__ == "__main__":
    main()
