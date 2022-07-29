import argparse

import edet_tflite

parser = argparse.ArgumentParser(description="A demo with img_path argument")
parser.add_argument(
    "-path",
    "--img_path",
    help="Adds a possibility to specify img_path .",
    default="images/people.jpeg",
)
parser.add_argument(
    "-n",
    "--n_boxes",
    help="Adds a possibility to choose a number of object",
    default=3,
    type=int,
)

args = parser.parse_args()

bboxes, class_ids, confs, img = edet_tflite.img_preprocess(
    img_path=args.img_path, n_boxes=args.n_boxes
)

edet_tflite.show_image(img)
