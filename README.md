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
  BATCH_SIZE: 2

MODEL:
  PRETRAINED_WEIGHTS: null
  FREEZE_LAYERS: "0"
  BACKBONE: "mobilenet_v2"
  ASPECT_RATIOS: [0.5, 1.0, 2.0]

TRAINING:
  EPOCHS: 3
  BASE_LR: 0.001
  STRATIFICATION_RATES: false
  MOMENTUM: 0.09
  WEIGHT_DECAY: 0.001
  OPTIMIZER: "SGD"
  LR_SCHEDULER: "StepLR"
  SCHEDULER_STEP_SIZE: 3
  SCHEDULER_GAMMA: 0.1
  WARMUP_STEPS: 0

AUGMENTATION:
  HORIZONTAL_FLIP_PROB: 0.0

OUTPUT:
  OUTPUT_PATH: "output"

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

- `BACKBONE` (default: `"mobilenet_v2"`): The name of the backbone that should be used. Available backbones are "mobilenet_v2", "resnet101" and "squeezenet1_1"
  - **Type**: String
  - **Effect**: Changes the backbone of the model.

- `ASPECT_RATIOS` (default: `"[0.5, 1.0, 20]"`): A comma-separated list of aspect ratios.
  - **Type**: String
  - **Effect**: Changes the aspect ratios of the anchor generator used for the backbone.

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

- `LR_SCHEDULER` (default: `"StepLR"`): The learning rate scheduler to use during training.
  - **Type**: String
  - **Possible Values**: `"StepLR"`, `"ReduceLROnPlateau"`
  - **Effect**: Controls the adjustment of the learning rate during training to improve convergence and performance. Different schedulers adjust the learning rate in different ways based on the specified schedule.

- `SCHEDULER_STEP_SIZE` (default: `3`): Step size for the learning rate scheduler.
  - **Type**: Integer
  - **Effect**: Specifies the number of epochs between each learning rate decay step when using the `StepLR` scheduler. At each step, the learning rate is adjusted by multiplying with `SCHEDULER_GAMMA`.

- `SCHEDULER_GAMMA` (default: `0.1`): Multiplicative factor for learning rate decay.
  - **Type**: Float
  - **Effect**: Determines the factor by which the learning rate is multiplied at each step. For example, with `StepLR`, the learning rate is multiplied by `SCHEDULER_GAMMA` every `SCHEDULER_STEP_SIZE` epochs, effectively reducing the learning rate to help the model converge more smoothly.

- `WARMUP_STEPS` (default: `0`): Number of warmup steps for learning rate scheduling.
  - **Type**: Integer
  - **Effect**: Defines the number of steps or epochs at the beginning of training during which the learning rate is increased linearly from a low initial value to the initial learning rate specified by the optimizer. Warmup helps in stabilizing the training process, particularly at the start.

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

## 7.1. Optimizer Comparison

### Setting

**Target**: This experiment focused on evaluating the impact of different optimizers **SGD**, **Adam**, and **AdamW** on model performance. The choice of optimizer can significantly influence the training dynamics, convergence speed, and final model accuracy. Specifically, the experiment aimed to understand how each optimizer, with the same base learning rate and weight decay, affects the training outcome.

**Link**: [Kaggle: Experiment 1](https://www.kaggle.com/code/henicosa/dl4cv-individual-project?scriptVersionId=185793029)

### Optimizer Overview

**SGD (Stochastic Gradient Descent)**:
- **Overview**: SGD updates the parameters by following the negative gradient of the loss function. It is a straightforward optimization algorithm with the option to include momentum, which helps to accelerate convergence by navigating along the relevant direction and dampening oscillations.
- **Link**: [PyTorch SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)

**Adam (Adaptive Moment Estimation)**:
- **Overview**: Adam combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients, making it particularly effective for sparse gradients and noisy problems.
- **Link**: [PyTorch Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

**AdamW (Adam with Weight Decay Regularization)**:
- **Overview**: AdamW modifies the typical Adam optimizer by decoupling the weight decay from the gradient-based update, which helps in achieving better generalization by preventing the weight decay from being included in the momentum term.
- **Link**: [PyTorch AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)


### Shared Configuration

```
# config.yaml
DATA:
  ROOT: "/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR/TD-TSR/"

OUTPUT:
  OUTPUT_PATH: "/kaggle/working/"

TRAINING:
  BASE_LR: 0.001
  WEIGHT_DECAY: 0.01
  MOMENTUM: 0.9
  EPOCHS: 7
```

### Results

![Plot](doc/exp1.png)

| Property      | SGD   | Adam  | AdamW |
| :------------ | :---- | :---- | :---- |
| Learning Rate | 0.001 | 0.001 | 0.001 |
| Weight Decay  | 0.01  | 0.01  | 0.01  |
| Momentum      | 0.9   | -     | -     |
| Final mAP     | 0.071 | 0.458 | 0.521 |


The results demonstrated that both Adam and AdamW significantly outperformed SGD in terms of model performance. As expercted, AdamW showed a slight edge over Adam in achieving better accuracy and stability during training. In contrast, SGD, despite the momentum parameter, lagged behind the adaptive optimizers.

### Interpretation

The experiment indicates that Adam and AdamW are more effective than SGD under the given training conditions, primarily due to their adaptive learning rate mechanisms, which help in faster and more stable convergence. However, the comparison might be slightly unfair to SGD, as the optimizer's sensitivity to weight decay differs from that of Adam and AdamW. The superior performance of AdamW over Adam suggests that decoupling weight decay from the learning rate, as implemented in AdamW, can lead to better generalization and overall performance.

## 7.2. AdamW Weight Decay

- No significant results

## 7.3. Backbone Comparison

### Setting

**Target**: This experiment aimed to evaluate the performance of different neural network backbones—MobileNetV2, SqueezeNet 1.1, and ResNet101—in a classification task. The backbone architecture can greatly influence the feature extraction capabilities and overall model accuracy. ResNet101 was connected to the model by extracting the [convolution feature map](https://stackoverflow.com/questions/58362892/resnet-18-as-backbone-in-faster-r-cnn).

**Link**: [Kaggle: Experiment 3](https://www.kaggle.com/code/henicosa/dl4cv-individual-project?scriptVersionId=186043012)

### Backbone Overview

**MobileNetV2**:
- **Overview**: MobileNetV2 is a lightweight convolutional neural network designed for mobile and resource-constrained environments. It uses depthwise separable convolutions and an inverted residual structure to achieve high efficiency and performance.
- **Link**: [PyTorch MobileNetV2](https://pytorch.org/vision/stable/models/mobilenetv2.html)

**SqueezeNet 1.1**:
- **Overview**: SqueezeNet is designed to achieve AlexNet-level accuracy with 50x fewer parameters, making it highly efficient. It uses Fire modules, which consist of squeeze and expand layers, to reduce the number of parameters.
- **Link**: [PyTorch SqueezeNet](https://pytorch.org/vision/stable/models/squeezenet.html)

**ResNet101**:
- **Overview**: ResNet101 is a deep convolutional neural network with 101 layers, utilizing residual connections to ease the training of very deep networks. These residual connections help mitigate the vanishing gradient problem, enabling the network to learn more complex features.
- **Link**: [PyTorch ResNet101](https://pytorch.org/vision/stable/models/resnet.html)

### Shared Configuration

```
# config.yaml
DATA:
  ROOT: "/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR/TD-TSR/"

OUTPUT:
  OUTPUT_PATH: "/kaggle/working/"

TRAINING:
  OPTIMIZER: "AdamW"
  BASE_LR: 0.001
  WEIGHT_DECAY: 0.015
  EPOCHS: 15
```

### Results

![Plot](doc/exp3.png)

| Property      | Mobilenet v2   | Squeezenet 1.1  | AdamW |
| :------------ | :---- | :---- | :---- |
| Final mAP     | 0.535 | 0.394 | 0.573 |
| Training Time | 1:26 h |  1:04 h | 2:10 h |

The results indicated that SqueezeNet 1.1 scored significantly lower compared to both MobileNetV2 and ResNet101. Among the latter two, ResNet101 demonstrated a slight performance advantage over MobileNetV2.

### Interpretation

The inferior performance of SqueezeNet 1.1 can be attributed to its design, which focuses on reducing the model size and computational cost, potentially at the expense of accuracy. On the other hand, MobileNetV2 and ResNet101 are more robust in feature extraction due to their more complex architectures.

The depth of ResNet101 allows it to learn more abstract representations, which generally contribute to better accuracy, albeit with increased computational requirements compared to MobileNetV2.

## 7.4. Aspect Ratios

### Setting

**Target**: This experiment aimed to assess the impact of changing the aspect ratios of the anchor generator on the performance of a detection model. The hypothesis was that aspect ratios resulting in longer boxes would better fit the objects in the dataset, which involve identifying rows and columns.

**Link**: [Kaggle: Your Home for Data Science](https://www.kaggle.com/code/henicosa/dl4cv-individual-project?scriptVersionId=186285151)

### Shared Configuration

```
# config.yaml
DATA:
  ROOT: "/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR/TD-TSR/"

OUTPUT:
  OUTPUT_PATH: "/kaggle/working/"

MODEL:
  BACKBONE: "resnet101"

TRAINING:
  OPTIMIZER: "AdamW"
  BASE_LR: 0.001
  WEIGHT_DECAY: 0.01
  EPOCHS: 7
```

### Results

![Plot](doc/exp_4.png)


| Property  | [0.5, 1, 2] | [0.33, 1, 3] | [0.33, 0.5, 1, 2, 3] |
| :-------- | :---------- | :----------- | :------------------- |
| Final mAP | 0.557       | 0.56         | 0.52                 |


The results showed a very slight but negligible advantage for the aspect ratios resulting in longer boxes. However, when comparing the bounding box visualizations, the predicted boxes with the longer aspect ratios appeared to be more aligned with the rows and columns of the dataset.

### Interpretation

Although the quantitative performance improvement was minimal, the qualitative assessment of bounding box visualizations suggests that aspect ratios tailored to the specific shapes of the objects in the dataset (rows and columns) can lead to more accurate detections. The longer aspect ratios helped the predicted boxes to better match the actual objects, indicating that aspect ratio customization is a valuable consideration for datasets with specific object shapes.


# 8. License

Please note that no license was provided for the assignment. My contributions to this project are licensed under the MIT License.

# 9. Disclaimer: AI Usage

Following generative AI tools were used to solve the assignment:

- ChatGPT
- Github Copilot

Please note, that because of the non-transparent data gathering in the training process of these tools, code might be generated and included that is incompatible with the MIT license.

