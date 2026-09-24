import os

import torch

from torch.utils.data import (
    DataLoader,
    random_split
)

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

from dataset import CustomObjectDataset

IMAGE_DIR = "dataset/images"

ANNOTATION_DIR = "dataset/annotations"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "screw_detector.pth"
)

NUM_CLASSES = 2
# 0 = Background
# 1 = Screw

BATCH_SIZE = 2

EPOCHS = 2

LEARNING_RATE = 0.005

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

dataset = CustomObjectDataset(
    IMAGE_DIR,
    ANNOTATION_DIR
)


print()
print(
    "Total images:",
    len(dataset)
)

train_size = int(
    0.80 * len(dataset)
)

validation_size = (
    len(dataset) - train_size
)


train_dataset, validation_dataset = (
    random_split(
        dataset,
        [train_size, validation_size]
    )
)


print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(validation_dataset)
)

def collate_fn(batch):

    return tuple(
        zip(*batch)
    )


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

print()
print(
    "Loading pretrained Faster R-CNN..."
)


weights = (
    FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)


model = fasterrcnn_resnet50_fpn(
    weights=weights
)

in_features = (
    model.roi_heads
    .box_predictor
    .cls_score
    .in_features
)


model.roi_heads.box_predictor = (
    FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print(
    "Using device:",
    device
)


model.to(device)

parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]


optimizer = torch.optim.SGD(
    parameters,
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=8,
    gamma=0.1
)

print(
    "STARTING TRAINING"
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0


    for batch_number, (
        images,
        targets
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = [
            image.to(device)
            for image in images
        ]


        targets = [

            {
                key: value.to(device)
                for key, value in target.items()
            }

            for target in targets
        ]

        loss_dict = model(
            images,
            targets
        )


        loss = sum(
            loss_value
            for loss_value in loss_dict.values()
        )


        optimizer.zero_grad()


        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
        )


        print(
            f"\rEpoch "
            f"{epoch + 1}/{EPOCHS} "
            f"| Batch "
            f"{batch_number}/{len(train_loader)} "
            f"| Loss: "
            f"{loss.item():.4f}",
            end=""
        )

    average_loss = (
        total_loss /
        len(train_loader)
    )


    scheduler.step()


    print()

    print(
        f"Epoch "
        f"{epoch + 1}/{EPOCHS} "
        f"completed "
        f"| Average Loss: "
        f"{average_loss:.4f}"
    )

    print()

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(
    "TRAINING COMPLETED"
)

print()
print(
    "Model saved to:"
)

print(
    MODEL_PATH
)