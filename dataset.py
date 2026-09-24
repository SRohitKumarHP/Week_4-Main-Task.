import os
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F


class CustomObjectDataset(Dataset):

    def __init__(self, image_dir, annotation_dir):

        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        self.images = sorted([
            file
            for file in os.listdir(image_dir)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        # ------------------------------------------
        # Image filename
        # ------------------------------------------

        image_name = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )


        # ------------------------------------------
        # Annotation filename
        # ------------------------------------------

        annotation_name = (
            os.path.splitext(image_name)[0]
            + ".json"
        )

        annotation_path = os.path.join(
            self.annotation_dir,
            annotation_name
        )


        # ------------------------------------------
        # Load image
        # ------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # ------------------------------------------
        # Load annotation
        # ------------------------------------------

        with open(
            annotation_path,
            "r"
        ) as file:

            annotation = json.load(file)


        # ------------------------------------------
        # Convert image to Tensor
        # ------------------------------------------

        image = F.to_tensor(
            image
        )


        # ------------------------------------------
        # Create bounding boxes
        # ------------------------------------------

        boxes = []
        labels = []


        for obj in annotation["objects"]:

            x = obj["x"]
            y = obj["y"]

            width = obj["width"]
            height = obj["height"]


            x2 = x + width
            y2 = y + height


            boxes.append([
                x,
                y,
                x2,
                y2
            ])


            # Class 1 = Screw
            labels.append(1)


        # ------------------------------------------
        # Convert to tensors
        # ------------------------------------------

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.tensor(
            labels,
            dtype=torch.int64
        )


        # ------------------------------------------
        # Area of bounding boxes
        # ------------------------------------------

        area = (
            (boxes[:, 2] - boxes[:, 0])
            *
            (boxes[:, 3] - boxes[:, 1])
        )


        # ------------------------------------------
        # Image ID
        # ------------------------------------------

        image_id = torch.tensor(
            [index]
        )


        # ------------------------------------------
        # Is crowd
        # ------------------------------------------

        iscrowd = torch.zeros(
            (len(boxes),),
            dtype=torch.int64
        )


        # ------------------------------------------
        # Target dictionary
        # ------------------------------------------

        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": image_id,

            "area": area,

            "iscrowd": iscrowd
        }


        return image, target