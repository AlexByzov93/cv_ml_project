This project is dedicated to different forms of implementation of EfficientDet network for object detection via TensorFlow Lite [see paper here](https://arxiv.org/abs/1911.09070).

# Task 2.2

This task was dedicated to adding CI/CD processes into a project. I added a new file `ci-testing.yml`, which does several things on each push or pull requests:

1. Sets up a clean Python environment
2. Installs pre-commits
3. Checks all files with pre-commits
4. Installs poetry
5. Uses pytest with poetry
6. Builds `.whl` and `.tar.gz` files with poetry
7. Saves artifacts of the build (accessible on Actions Page for the last workflow runs)

To use this workload, you just need to do the same steps as in task 2.1 (read below)

## Pytest

In this task we needed to create two tests:

1. A regression test using a single image that checks results are always the same
2. A “no error” test with big image, small image

Since I am adding testing into the project, it will be more fruitful to refactor `demo.py` into different functions, which allows to test their work separately. Moreover, I also added a `main` function that does the same thing as previous poetry file.

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