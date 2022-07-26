This project is dedicated to different forms of implementation of EfficientDet network for object detection via TensorFlow Lite [see paper here](https://arxiv.org/abs/1911.09070).

# Task 2.2

This task was dedicated to adding CI/CD processes into a project. I added a new file `ci-testing.yml`, which does several things on each push or pull requests:

1. Sets up a clean Python environment
2. Installs pre-commits
3. Checks all files with pre-commits
4. Installs poetry
5. Uses pytest with poetry
6. Builds `.whl` and `.tar.gz` files with poetry
7. Saves artifacts of the build


 adding style and code linting, and pre-commit hooks. I continued to use `Poetry` for solving this task. In this instruction I describe how to setup the project for local development

## Step 1. Install Poetry

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

## Step 2. Clone repository

```bash
git clone https://github.com/AlexByzov93/cv_ml_project.git@task-2_1

cd cv_ml_project
```

## Step 3. Install package

```bash
poetry install
poetry shell
```

## Step 4. Install pre-commit hooks

```bash
pre-commit install
```

Now the project is ready for local development, and will check files on each commit for code linting, style linting, etc. If you want to check the files before commiting them, you can use the following command:

```bash
pre-commit run --all-files
```