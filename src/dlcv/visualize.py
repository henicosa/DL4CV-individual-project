import argparse
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from math import ceil

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = "tight"

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def visualize_bbox_output(img, output, device):
    colors = ["white", "red", "magenta", "green", "orange", "cyan", "fuchsia"]
    img = img.to(device)
    img = img.mul(255).byte()
    drawn_boxes = draw_bounding_boxes(
        image=img,
        boxes=output["boxes"].cpu().detach(),
        colors=[colors[int(label)] for label in output['labels']],
        labels=[str(label.item()) for label in output['labels']],
        width=5
    )
    return drawn_boxes.cpu()

def generate_visualisations(data_loader, number, folder, model, device):
    iterations = 0
    columns = 3
    rows = int(ceil(number / columns))
    
    # Set the figure size to fit portrait images and the notebook width
    fig, axs = plt.subplots(rows, columns, figsize=(11, rows * 5))
    axs = axs.flatten() 

    for images, targets in data_loader:
        if iterations < number:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for idx, output in enumerate(outputs):
                tensor = visualize_bbox_output(images[idx], output, device)
                pil_image = to_pil_image(tensor)
                axs[iterations].imshow(pil_image)
                axs[iterations].axis("off")      
                pil_image.save(f"{folder}BBOX_{targets[idx]['image_name']}.png", "PNG")
                iterations += 1

                if iterations >= number:
                    break

        if iterations >= number:
            break

    plt.tight_layout()
    plt.show()


import os
import csv
def plot_mAP(mAP_values, path, run_name):
    """
    Plots the Mean Average Precision (mAP) values over the evaluation epochs.
    
    Args:
        mAP_values (list): List of mAP values over the evaluation epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mAP_values, marker='o', linestyle='-', color='b', label='mAP')
    plt.xlabel('Evaluation Epochs')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision (mAP) Over Evaluation Epochs')
    plt.legend()
    plt.grid(True)
    # export mAP_values with epoch and value as csv
    data = [{"epoch": i + 1, "mAP": value} for i, value in enumerate(mAP_values)]
    header = ["epoch", "mAP"]
    with open(os.path.join(path, run_name + ".csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        
        writer.writeheader()  # Write the header
        writer.writerows(data)  # Write the data rows

    plt.savefig(os.path.join(path, run_name + ".png"), format="png")
    plt.show()


