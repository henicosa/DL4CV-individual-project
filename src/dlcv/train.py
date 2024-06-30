import argparse
from dlcv.dataset import CisolTDTSRDataset
import json
from dlcv.visualize import plot_mAP, generate_visualisations

# This package internal functions should be used here
from dlcv.models import get_model
import torch
from dlcv.utils import *
from dlcv.detection.engine import train_one_epoch, evaluate
import os
from dlcv.schedule import WarmupThenScheduler


def train_notebook(run_name, args):

    argv = [run_name]
    # iterate over the dictionary and add the key and value to the argv list
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key.lower())
        else:
            argv.append("--" + key.lower())
            argv.append(str(value))

    # call the main function with the argv list
    try: 
        main(parse_args(argv))
    except SystemExit as e:
        print(argv)
        # print the error message
        print("Error in the arguments")
        print(e)


def parse_list(arglist):
    arglist = "".join(arglist)
    arglist = arglist.replace("[","")
    arglist = arglist.replace("]","")
    arglist = arglist.split(",")
    arglist = [float(arg) for arg in arglist]
    return arglist

from torchvision.transforms import v2 as T

from torch.cuda import empty_cache

def clear_memory():
    empty_cache()

def main(args):
    
    # Define transformations for training and testing

    def get_transform(train):
        transforms = []
        if train:
            if args.horizontal_flip_prob:
                transforms.append(T.RandomHorizontalFlip(args.horizontal_flip_prob))
        transforms.append(T.ToDtype(torch.float, scale=True))
        return T.Compose(transforms)
    
    # Set device
    device = (
        "cuda"
        if (torch.cuda.is_available() and not args.no_cuda)
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # 5 classes in dataset + background
    num_classes = 6
    # use our dataset and defined transformations
    dataset_train = CisolTDTSRDataset(args.root, "train", get_transform(train=True))
    dataset_val = CisolTDTSRDataset(args.root, "validate", get_transform(train=False))
    dataset_test = CisolTDTSRDataset(args.root, "test", get_transform(train=False))
    output_path = args.output_path

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset_train.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset_val.collate_fn
    )


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset_test.collate_fn
    )

    chosen_backbone = args.backbone.lower()

    if args.aspect_ratios:
        aspect_ratios = parse_list(args.aspect_ratios)
        aspect_ratios = tuple(aspect_ratios)
    else:
        aspect_ratios = (0.5, 1.0, 2.0)

    # get the model using our helper function
    model = get_model(num_classes, chosen_backbone, aspect_ratios)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Load pretrained weights if specified
    if args.pretrained_weights:
        model = load_pretrained_weights(model, args.pretrained_weights, device)

    # Freeze layers if set as argument
    if args.freeze_layers:
        freeze_layers(model, args.freeze_layers.split(","))

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    chosen_optimizer = args.optimizer.lower()

    if chosen_optimizer == "adam":
        # Optimizer: Adam
        optimizer = torch.optim.Adam(
            params,
            lr=args.base_lr,
            weight_decay=args.weight_decay
        )

    elif chosen_optimizer == "adamw":
        # Optimizer: AdamW
        optimizer = torch.optim.AdamW(
            params,
            lr=args.base_lr,
            weight_decay=args.weight_decay
        )

    else:
        optimizer = torch.optim.SGD(
            params,
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

    if args.lr_scheduler == "StepLR":
        # Create a learning rate scheduler
        post_warmup_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )

    elif args.lr_scheduler == "ReduceLROnPlateau":
        # Create a learning rate scheduler
        post_warmup_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_gamma,
            patience=args.scheduler_step_size,
            verbose=True
        )


    lr_scheduler = WarmupThenScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        post_warmup_scheduler=post_warmup_scheduler
    )

    num_epochs = args.epochs
    mAPs = []

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        clear_memory()
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        
        # evaluate on the test dataset
        cocoeval = evaluate(model, data_loader_val, device=device)
        metric = cocoeval.coco_eval["bbox"].stats[0]
        mAPs.append(metric)

        # update the learning rate
        if args.lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler.step(metric)
        else:
            lr_scheduler.step()

    plot_mAP(mAPs, os.path.join(output_path, "performance"), args.run_name)

    if output_path:
        save_model(model, os.path.join(args.output_path, "models/" + args.run_name))

        predictions = []
        model.eval()

        # Iterate over the test dataset and make predictions
        for images, target in data_loader_test:

            images = list(img.to(device) for img in images)
            outputs = model(images)

            # Process detections from model output
            for idx, output in enumerate(outputs):
                for i in range(len(output["labels"])):
                    prediction = {}
                    prediction["file_name"] = target[0]["image_name"]  # Assuming you have a way to get image file names
                    prediction["category_id"] = int(output["labels"][i])  # Assuming labels start from 1

                    x1 , y1, x2, y2 = output["boxes"][i].cpu().detach().numpy().tolist()  # Convert bbox to list
                    prediction["bbox"] = [x1, y1, x2-x1, y2-y1]
                    #print(x1,y1,x2,y2)
                    prediction["score"] = float(output["scores"][i])  # Convert score to float
                    predictions.append(prediction)

        # Save predictions to a JSON file
        with open(os.path.join(output_path, "predictions/" + args.run_name + "_prediction.json"), "w") as f:
            json.dump(predictions, f, indent=4)

        image_path = os.path.join(output_path, args.run_name)
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        generate_visualisations(data_loader_test, 3, image_path, model, device)

    # Save results to CSV
    # write_results_to_csv(args.results_csv + "/" + args.run_name, train_losses, test_losses, test_accuracies)

    # Save the model using the default folder



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the Customizable Network")

    # add positional argument for the run name
    parser.add_argument("run_name", type=str, help="Give your training a run name")

    # Add arguments
    parser.add_argument("--root", type=str, default="data", help="Root directory of the dataset")
    parser.add_argument("--subset_size", type=str, default=None, help="Number of samples to use from the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--conv_layers", type=int, default=2, help="Number of convolutional layers", choices=[2,3])
    parser.add_argument("--filters_conv1", type=int, default=16, help="Number of filters in the first conv layer")
    parser.add_argument("--filters_conv2", type=int, default=32, help="Number of filters in the second conv layer")
    parser.add_argument("--filters_conv3", type=int, default=0, help="Number of filters in the third conv layer")
    parser.add_argument("--dense_units", type=int, default=128, help="Number of units in the dense layer")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to pretrained weights file")
    parser.add_argument("--freeze_layers", type=str, default=0, help="Comma-separated list of layer names to freeze")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2", help="Backbone for the model")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base learning rate for the optimizer")
    parser.add_argument("--aspect_ratios", type=list, default=[0.5,1,2], help="Aspect ratios for anchor generator")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for the optimizer")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Choose between 'SGD', 'Adam' and 'AdamW' optimizers")
    parser.add_argument("--stratification_rates", type=str, default=None, help="Dictionary of stratification rates")
    parser.add_argument("--scheduler_step_size", type=int, default=3, help="Step size for the learning rate scheduler")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Gamma for the learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--horizontal_flip_prob", type=float, default=0.0, help="Probability of applying horizontal flip; 0 means no horizontal flip")
    parser.add_argument("--rotation_degrees", type=float, default=0.0, help="Max degrees to rotate; 0 means no rotation")
    parser.add_argument("--results_csv", type=str, default="results", help="Directory to save the CSV file of training results")
    parser.add_argument("--save_model_path", type=str, default="saved_models", help="Directory to save the trained model")
    parser.add_argument("--output_path", type=str, default="output", help="Directory to save all outputs")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--do_early_stopping", action="store_true", help="Turn early stopping on or off")

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    # You can find examples for argparse in tools/plot.py and tools/visualize.py
