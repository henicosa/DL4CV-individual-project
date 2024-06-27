[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/GVYBQ-r3)
This is a suggested outline. You are free in your design. You will need to create the project structure and move the file `model.py` to its proper location.

# 1. Installation Instructions

## Kaggle

```python
token = "github_pat_secret"
user = "BUW-CV"
repo_name = "dlcv24-assignment-4-henicosa"
url = f"https://{user}:{token}@github.com/{user}/{repo_name}.git"
!pip install git+{url}
```

To recieve a token contact me.

## Local Package

### Prerequisites

Ensure you have Python 3.10 installed. You can download it from the [official Python website](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/yourusername/dlcv-project.git
cd dlcv-project
```

### Using Virtual Environment (Recommended)

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install the Package

Install the package in interactive mode directly from the `setup.py` file:
```bash
pip install -e .
```

# 2. Link to Single Kaggle Notebook

- https://www.kaggle.com/code/henicosa/dl4cv-ass4/edit

# 3. Resources

- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
- https://github.com/Torky24/TexBig---Object-Detection/tree/main

# 4. How to run a training

Running a training will automatically store the used set of hyperparameters as a `<run_name>.yaml` in the models directory. It will also store the model as `<run_name>.pth` in the same directory and the accuracy in each epoch as `<run_name>.csv` int the results directory.

## With a configuration dictionary

Use the function `execute_training(run_name, options)` to initiate the training with a run_name (string) and options as the configuration dictionary which can contain any command line option (without "--" prefix) as a key and the positional argument as a value. To initiate the training with default options you can pass an empty dictionary.

## From a configuration file

You can also pass the path to an existing configuration file to the function `execute_training_from_config_file(run_name, filepath)` to initate a training. The format of the file is decribed in the next section.

# 5. How to configure different trainings

This is the default configuration in the YAML format. You can include any option in your own configuration file for a training.

```yaml
# config.yaml
DATA:
  ROOT: "data"
  BATCH_SIZE: 64

MODEL:
  PRETRAINED_WEIGHTS: null
  FREEZE_LAYERS: "0"

TRAINING:
  EPOCHS: 3
  BASE_LR: 0.001
  DO_EARLY_STOPPING: false

AUGMENTATION:
  HORIZONTAL_FLIP_PROB: 0.0

OUTPUT:
  RESULTS_CSV: "results"
  SAVE_MODEL_PATH: "saved_models"

SYSTEM:
  NO_CUDA: false
```

It is divided in six categories: Data, Model, Training, Augmentation, Output and System. The next section describes the configurable hyper-parametes in detail.

# 6. Overview of available configurable hyper-parameters

## Data

This section contains hyper-parameters related to reading and splitting the data.

- `DATA_ROOT` (default: `data`): The root directory of the dataset.

- `BATCH_SIZE` (default: `64`): The batch size for training and evaluation.

## Model

This section contains hyper-parameters related to the model's architecture and initialization.

- `PRETRAINED_WEIGHtS` (default: `None`): Path to a file with pretrained weights to initialize the model. If not specified, the model will start with random weights.

- `FREEZE_LAYERS` (default: `0`): A comma-separated list of layer names to freeze during training. This means these layers' weights will not be updated.

## Model

This section contains hyper-parameters influencing only the training process.


- `EPOCHS` (default: `3`): The number of epochs to train the model.

- `BASE_LR` (default: `0.001`): The base learning rate for the optimizer.

## Model

This section contains hyper-parameters for preparing the data before the training process.

- `HORIZONTAL_FLIP_PROB` (default: `0.0`): The probability of applying a horizontal flip to images during training for data augmentation. A value of 0 means no horizontal flip.

## Output

This section controls the output path values.

- `RESULTS_CSV` (default: `results`): The directory to save the CSV file containing training results.

- `SAVE_MODEL_PATH` (default: `saved_models`): The directory to save the trained model.

## System

This section contains hyper-parameters controlling how the host system is used during training.

- `NO_CUDA` (default: `False`): Disable CUDA even if available, forcing the use of CPU.


# 7. Visualisations

After each run, a visualisation of the mean average precision over the epochs is plotted.

![Plot](doc/training_graph.png)

After each run, two images with the detected bounding boxes are generated from the training set for reference. They are also saved in the model directory. Each category produces differently colored bounding boxes.

![Plot](doc/bboxes.png)

# 8. Succesfull Training

A succesfull training run was submitted with a mAP of over 40% was submitted under the university username `viji5369` on the evalAI CISOL leaderboard.

# License

Please note that no license was provided for the assignment. My contributions to this project are licensed under the MIT License.

# Disclaimer: AI Usage

Following generative AI tools were used to solve the assignment:

- ChatGPT
- Github Copilot

Please note, that because of the non-transparent data gathering in the training process of these tools, code might be generated and included that is incompatible with the MIT license.

