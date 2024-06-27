# 1. Installation Instructions

## Kaggle

```python
token = "github_pat_secret"
user = "BUW-CV"
repo_name = "dlcv24-individual-final-project-henicosa"
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

- https://www.kaggle.com/code/henicosa/dl4cv-individual-project/edit/


# 3. How to run a training

Running a training will automatically store the used set of hyperparameters as a `<run_name>.yaml` in the models directory. It will also store the model as `<run_name>.pth` in the same directory and the accuracy in each epoch as `<run_name>.csv` int the results directory.

## With a configuration dictionary

Use the function `execute_training(run_name, options)` to initiate the training with a run_name (string) and options as the configuration dictionary which can contain any command line option (without "--" prefix) as a key and the positional argument as a value. To initiate the training with default options you can pass an empty dictionary.

## From a configuration file

You can also pass the path to an existing configuration file to the function `execute_training_from_config_file(run_name, filepath)` to initate a training. The format of the file is decribed in the next section.

# 4. How to configure different trainings

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

# 5. Overview of available configurable hyper-parameters

Below, each configuration option is explained in detail along with possible input values and their effects.

## DATA

This section contains hyper-parameters related to the dataset and batch processing.

- `ROOT` (default: `"data"`): Path to the root directory where the dataset is stored.
  - **Type**: String
  - **Effect**: Specifies the directory containing the dataset to be used for training and evaluation.

- `BATCH_SIZE` (default: `2`): Number of samples per batch during training.
  - **Type**: Integer
  - **Effect**: Controls the number of images processed in parallel during each training iteration. Larger batch sizes may improve training stability but require more memory.

## MODEL

This section contains hyper-parameters related to the model's architecture and initialization.

- `PRETRAINED_WEIGHTS` (default: `false`): Whether to load pretrained weights for the model.
  - **Type**: Boolean
  - **Effect**: If `true`, the model will be initialized with weights pretrained on a large dataset (e.g., ImageNet), which can speed up convergence and improve performance.

- `FREEZE_LAYERS` (default: `"0"`): A comma-separated list of layer names to freeze during training. This means these layers' weights will not be updated.
  - **Type**: String
  - **Effect**: Freezes the specified layers, preventing their weights from being updated during training. Useful for transfer learning to retain pretrained features.

## TRAINING

This section contains hyper-parameters related to the training process.

- `EPOCHS` (default: `3`): Number of epochs to train the model.
  - **Type**: Integer
  - **Effect**: Determines how many times the entire dataset is passed through the model during training. More epochs can lead to better performance but increase training time.

- `BASE_LR` (default: `0.001`): Base learning rate for the optimizer.
  - **Type**: Float
  - **Effect**: Controls the step size at each iteration while moving towards a minimum of the loss function. A lower value makes the training process slower and more stable, while a higher value speeds up training but may cause instability.

- `STRATIFICATION_RATES` (default: `false`): Whether to use stratified sampling during training.
  - **Type**: Boolean
  - **Effect**: If `true`, ensures that each batch has a balanced representation of different classes, which can be important for imbalanced datasets.

- `MOMENTUM` (default: `0.09`): Momentum factor for the optimizer.
  - **Type**: Float
  - **Effect**: Accelerates gradient vectors in the correct direction, leading to faster convergence. Only used in combination with SGD optimizer.

- `WEIGHT_DECAY` (default: `0.001`): Weight decay (L2 penalty) for the optimizer.
  - **Type**: Float
  - **Effect**: Prevents overfitting by penalizing large weights in the model, effectively acting as regularization.

- `OPTIMIZER` (default: `"SGD"`): The type of optimizer to use for training.
  - **Type**: String
  - **Possible Values**: `"SGD"`, `"Adam"`, `"AdamW"`
  - **Effect**: Determines the optimization algorithm used for updating model weights. Different optimizers have various convergence properties and performance characteristics.

- `BACKBONE` (default: `"mobilenet_v2"`): The backbone network to be used in the model.
  - **Type**: String
  - **Possible Values**: `"mobilenet_v2"`, `"resnet50"`, `"resnet101"`
  - **Effect**: The backbone network extracts features from images. Choosing a different backbone can affect the model’s performance and computational requirements.

## AUGMENTATION

This section contains hyper-parameters related to data augmentation techniques applied during training.

- `HORIZONTAL_FLIP_PROB` (default: `0.0`): Probability of applying horizontal flip augmentation.
  - **Type**: Float
  - **Effect**: Randomly flips images horizontally during training with the specified probability, helping to generalize the model to different orientations.


## OUTPUT

This section contains hyper-parameters related to the output of the training process.

- `OUTPUT_PATH` (default: `"output"`): Directory where output files (e.g., models, logs) will be saved.
  - **Type**: String
  - **Effect**: Specifies the path for saving training artifacts, including trained models and performance logs.

## SYSTEM

This section contains hyper-parameters related to system settings and configurations.

- `NO_CUDA` (default: `false`): Whether to disable CUDA (GPU) usage.
  - **Type**: Boolean
  - **Effect**: If `true`, forces the use of CPU even if a GPU is available. Useful for debugging or running on systems without GPUs.

- `DO_EARLY_STOPPING` (default: `false`): Whether to apply early stopping during training.
  - **Type**: Boolean
  - **Effect**: If `true`, training will be stopped early if the model’s performance on the validation set does not improve for a specified number of epochs, preventing overfitting and saving time.



# 6. Visualisations

After each run, a visualisation of the mean average precision over the epochs is plotted and saved in the perfomance directory together with the raw data in the csv format.

![Plot](doc/AdamW.png)

After each run, three images with the detected bounding boxes are generated from the training set for reference. They are also saved in the output directory. 

![Plot](doc/bbox.png)

Please refer to this table to attribute the category to the bounding box displayed in the visualisation.

| ID  | Name   | Color   |
| :-- | :----- | :------ |
| 1   | table  | red     |
| 2   | row    | magenta |
| 3   | cell   | green   |
| 4   | header | orange  |
| 5   | column | cyan    |


# 7. Experiments

## 7.1. Optimizer

- [Link to Experiment 1](https://www.kaggle.com/code/henicosa/dl4cv-individual-project?scriptVersionId=185793029)

In this experiment three different optimizers were compared: SDG, Adam and AdamW.

![Plot](doc/exp1.png)

The training run `AdamW` was submitted under the university username `viji5369` on the evalAI CISOL leaderboard and scored mAP = 50.16%.

# 8. License

Please note that no license was provided for the assignment. My contributions to this project are licensed under the MIT License.

# 9. Disclaimer: AI Usage

Following generative AI tools were used to solve the assignment:

- ChatGPT
- Github Copilot

Please note, that because of the non-transparent data gathering in the training process of these tools, code might be generated and included that is incompatible with the MIT license.

