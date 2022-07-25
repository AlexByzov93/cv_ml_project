This project is dedicated to different forms of implementation of EfficientDet network for object detection via TensorFlow Lite [see paper here](https://arxiv.org/abs/1911.09070).

# Task 1.2.

In this task I package my code with `Poetry`, which allows for a better dependency management, but the goal is still the same - implement EfficientDet network for object detection and show it with `demo` code.

This task requires us to implement two functionalities:

1. installation of the package from something similar to `python3 -m build`
2. installation of the package from GitHub repository

## Installation from something similar to `build` functionality

### Step 1. Install Poetry

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

### Step 2. Clone repository

```bash
https://github.com/AlexByzov93/cv_ml_project.git

cd cv_ml_project
```

### Step 3. Install package

```bash
poetry install # Installs poetry and allows to use it the functionality of the demo.py
```

### Step 4. Start using Poetry shell (similar to activated virtualenv)

```bash
cd efdet_tfl

poetry shell # similar to source venv/bin/activate
```

### Step 5. Test the script

```bash
python3 demo.py
```

## Installation from GitHub

Just for example, let me show how to combine virtual environments with installation from GitHub of `Poetry` package

### Step 1. Create and activatevirtual environment

```bash
python3 -m venv poetry_package

source poetry_package/bin/activate
```

### Step 2. Install package from git

```bash
pip3 install --upgrade pip
pip3 install https://github.com/AlexByzov93/cv_ml_project.git@task-1_2
```

### Step 3. Add image and weights into the folder

```bash
mkdir images weights # creates empty folders
wget -O images/dogs.jpeg https://github.com/AlexByzov93/cv_ml_project/raw/task-1_1/images/dog.jpeg # downlods image of a dog
wget -O weights/lite-model_efficientdet_lite0_detection_default_1.tflite
https://github.com/AlexByzov93/cv_ml_project/raw/task-1_1/weights/lite-model_efficientdet_lite0_detection_default_1.tflite # downloads weights of the model
```

### Step 4. Test the package

```bash
python3 # starts python3 within virtual environment
from efdet_tfl import demo #imports demo file from efdet_tfl package
demo.show_demo() 
```

# Troubleshooting

You might have a problem with cv2 library in python, if you work with Ubuntu Linux Server, because it does not have `libsglu1` file. To solve this problem, you can use the following bash command:

```bash
sudo apt install libglu1-mesa-dev
```

This and many other problems could be easily overcomed with Docker image.