import csv
import torch
import numpy as np
import seaborn as sns
import torchvision.transforms
import matplotlib.pyplot as plt

def load_pretrained_weights(network, weights_path, device):
    """
    Loads pretrained weights (state_dict) into the specified network.

    Args:
        network (nn.Module): The network into which the weights are to be loaded.
        weights_path (str or pathlib.Path): The path to the file containing the pretrained weights.
        device (torch.device): The device on which the network is running (e.g., 'cpu' or 'cuda').
    Returns:
        network (nn.Module): The network with the pretrained weights loaded and adjusted if necessary.
    """
    
    network.load_state_dict(torch.load(weights_path, map_location=device))
    return network


def freeze_layers(network, frozen_layers):
    """
    Freezes the specified layers of a network. Freezing a layer means its parameters will not be updated during training.

    Args:
        network (nn.Module): The neural network to modify.
        frozen_layers (list of str): A list of layer identifiers whose parameters should be frozen.
    """
    # Source: https://stackoverflow.com/questions/62523912/freeze-certain-layers-of-an-existing-model-in-pytorch
    for name, param in network.named_parameters():
        # freezes layerx.weight and layerx.bias
        if name.replace('.weight', '').replace('.bias', '') in frozen_layers:
            param.requires_grad = False

def save_model(model, path):
    """
    Saves the model state_dict to a specified file.

    Args:
        model (nn.Module): The PyTorch model to save. Only the state_dict should be saved.
        path (str): The path where to save the model. Without the postifix .pth
    """
    torch.save(model.state_dict(), path + ".pth")

def get_stratified_param_groups(network, base_lr=0.001, stratification_rates=None):
    """
    Creates parameter groups with different learning rates for different layers of the network.

    Args:
        network (nn.Module): The neural network for which the parameter groups are created.
        base_lr (float): The base learning rate for layers not specified in stratification_rates.
        stratification_rates (dict): A dictionary mapping layer names to specific learning rates.

    Returns:
        param_groups (list of dict): A list of parameter group dictionaries suitable for an optimizer.
                                     Outside of the function this param_groups variable can be used like:
                                     optimizer = torch.optim.Adam(param_groups)
    """
    param_groups = []
    for name, param in network.named_parameters():
        lr = stratification_rates.get(name.replace('.weight', '').replace('.bias', ''), base_lr)
        param_groups.append({'params': param, 'lr': lr})
    return param_groups

def get_transforms(train=True, horizontal_flip_prob=0.0, rotation_degrees=0.0):
    """
    Creates a torchvision transform pipeline for training and testing datasets. For training, augmentations
    such as horizontal flipping and random rotation can be included. For testing, only essential transformations
    like normalization and converting the image to a tensor are applied.

    Args:
        train (bool): Indicates whether the transform is for training or testing. If True, augmentations are applied.
        horizontal_flip_prob (float): Probability of applying a horizontal flip to the images. Effective only if train=True.
        rotation_degrees (float): The range of degrees for random rotation. Effective only if train=True.

    Returns:
        torchvision.transforms.Compose: Composed torchvision transforms for data preprocessing.
    """
    transforms = []
    transforms.append(torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR))
    if train:
        if horizontal_flip_prob != 0.0:
            transforms.append(torchvision.transforms.RandomHorizontalFlip(horizontal_flip_prob))
        if rotation_degrees != 0.0:
            transforms.append(torchvision.transforms.RandomRotation(rotation_degrees))
    transforms.append(torchvision.transforms.ToTensor())
    # Normalize the image with the mean and standard deviation based on DLCV_1_optimization_and_regularization.pdf slide 47
    transforms.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return torchvision.transforms.Compose(transforms)
    

def write_results_to_csv(file_path, train_losses, test_losses, test_accuracies):
    """
    Writes the training and testing results to a CSV file.

    Args:
        file_path (str): Path to the CSV file where results will be saved. Without the postfix .csv
        train_losses (list): List of training losses.
        test_losses (list): List of testing losses.
        test_accuracies (list): List of testing accuracies.
    """
    with open(file_path + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch], test_accuracies[epoch]])

def plot_multiple_losses_and_accuracies(model_data_list):
    """
    Plots training and testing losses and accuracies for multiple models.

    Args:
        model_data_list (list of dict): A list of dictionaries containing the following keys:
            - 'name' (str): The name of the model (for the legend)
            - 'train_losses' (list): Training losses per epoch
            - 'test_losses' (list): Testing losses per epoch
            - 'test_accuracies' (list): Testing accuracies per epoch
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for model_data in model_data_list:
        axs[0].plot(model_data['train_losses'], label=f"{model_data['name']} Train Loss")
        axs[0].plot(model_data['test_losses'], label=f"{model_data['name']} Test Loss")
        axs[1].plot(model_data['test_accuracies'], label=f"{model_data['name']} Test Accuracy")
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Testing Losses')
    axs[0].legend()
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Testing Accuracies')
    axs[1].legend()
    plt.tight_layout()
    
    # save the plot
    plt.savefig("results.png") 

    plt.show()
    

def plot_samples_with_predictions(images, labels, predictions, class_names):
    """
    Plots a grid of images with labels and predictions, with dynamically adjusted text placement.

    Args:
        images (Tensor): Batch of images.
        labels (Tensor): True labels corresponding to the images.
        predictions (Tensor): Predicted labels for the images.
        class_names (list): List of class names indexed according to labels.
    """
    num_images = images.shape[0]
    # get next next square number with an integer square root
    num_rows = int(np.ceil(np.sqrt(num_images)))

    # make plot of shape 8 by 8
    fig, axs = plt.subplots(num_rows, num_rows, figsize=(20, 20))
    for i in range(num_rows):
        for j in range(num_rows):
            idx = i * num_rows + j
            if idx < num_images:
                image = images[idx].permute(1, 2, 0).cpu().numpy()
                label = class_names[labels[idx].item()]
                prediction = class_names[predictions[idx].item()]
                axs[i, j].imshow(image)
                axs[i, j].axis('off')
                axs[i, j].set_title(f"Label: {label}\nPrediction: {prediction}", fontsize=8)
    plt.savefig("samples.png")
    plt.show()




def plot_confusion_matrix(labels, preds, class_names):
    """
    Plots a confusion matrix using ground truth labels and predictions.
    """
    # generate confusion matrix
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for label, pred in zip(labels, preds):
        cm[label, pred] += 1
    cm = cm / cm.sum(axis=1)[:, np.newaxis]

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()
