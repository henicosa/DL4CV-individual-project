import os
import json
import torch
from PIL import Image, ImageDraw
import numpy as np
from torchvision.datasets import CocoDetection
from torchvision.io import read_image
from torchvision import tv_tensors
import torchvision

class CisolTDTSRDataset(CocoDetection):
    def __init__(self, root: str, mode: str, transform=None):
        """
        Custom dataset for the CISOL TD-TSR dataset.

        Args:
            root (str): Root directory of the dataset.
            mode (str): Dataset mode. Supported modes are 'train', 'validate', and 'test'.
            transform (callable): Optional transform to be applied to the image and target.
        """
        super(torchvision.datasets.CocoDetection, self).__init__(root) 
        
        
        self.mode = mode
        if self.mode == "train":
            self.img_path = os.path.join(root, "images/train")
            self.json_path = os.path.join(root, "annotations/train.json")
        elif self.mode == "validate":
            self.img_path = os.path.join(root, "images/val")
            self.json_path = os.path.join(root, "annotations/val.json")
        elif self.mode == "test":
            self.img_path = os.path.join(root, "images/test")
        else:
            raise ValueError(
                "Invalid mode. Supported modes are 'train', 'validate', and 'test'."
            )
            

        if mode != "test":
            from pycocotools.coco import COCO
            self.coco = COCO(self.json_path)
            self.ids = list(sorted(self.coco.imgs.keys()))
            
            with open(self.json_path, "r") as json_file:
                self.data = json.load(json_file)
            
            self.annotations = self.data["annotations"]
            self.img = self.preprocess_data()
        else:
            self.data = {}
            self.img = create_image_array(self.img_path)
            
        self.transform = transform

    def __len__(self):
        if hasattr(self, "img"):
            return len(self.img)
        else:
            img_files = sorted(os.listdir(self.img_path))
            return len(img_files)

    def __getitem__(self, idx):
        if hasattr(self, "img"):
            img_name = self.img[idx]["file_name"]
            img_id = self.img[idx]["id"]
            img_path = os.path.join(self.img_path, img_name)
        else:
            img_files = sorted(os.listdir(self.img_path))
            img_name = img_files[idx]
            img_path = os.path.join(self.img_path, img_name)
            img_id = None

        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size
        img = read_image(img_path, mode = torchvision.io.ImageReadMode.RGB)
        
        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        #if self.mode == "test":
        #    target["img_name"] = img_name

        target = {}
        
        boxes = []
        labels = []
        areas = []
        iscrowds = []
            
        if hasattr(self, "annotations") and img_id is not None:
            
            # filter for all annotations that apply to the selected image
            annotations = [
                anno for anno in self.annotations if anno["image_id"] == img_id
            ]

            for annotation in annotations:
                xmin, ymin, width, height = annotation["bbox"]
                xmax = xmin + width
                ymax = ymin + height
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(annotation["category_id"])
                areas.append(annotation["area"])
                iscrowds.append(annotation["iscrowd"])

        # Convert boxes to tensor and clip to image size
        boxes = torch.as_tensor(boxes)#, dtype=torch.float32)

        """
        from torchvision.transforms.v2.functional import sanitize_bounding_boxes
        boxes = sanitize_bounding_boxes(
            bounding_boxes=boxes, canvas_size=(img_height, img_width), format="xyxy"
        )
        boxes = boxes[0]
        """

        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(labels) #, dtype=torch.int64),
        target["image_id"] = img_id
        target["image_name"] = img_name
        target["area"] = torch.as_tensor(areas) # dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowds) # dtype=torch.uint8)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def preprocess_data(self):
        """
        Preprocesses the dataset by removing images without annotations.

        Returns:
            list: Preprocessed image metadata.
        """
        images = []
        for img in self.data["images"]:
            img_id = img["id"]
            annotations = [
                anno for anno in self.data["annotations"] if anno["image_id"] == img_id
            ]
            if len(annotations) > 0:
            
                images.append(img)
        return images
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
def create_image_array(test_folder):
    image_array = []
    # List all files in the test folder
    files = os.listdir(test_folder)
    # Sort files by name to ensure consistent ordering
    files.sort()
    # Iterate over each file
    for idx, filename in enumerate(files, start=1):
        # Create a dictionary for each image
        if filename.endswith(".png"):
            image_dict = {
                "id": idx,  # Image ID starting from 1
                "file_name": filename  # File name of the image
            }
            # Append the dictionary to the image_array
            image_array.append(image_dict)
    
    return image_array