import cv2
import torch

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

from torchvision.transforms import functional as F

MODEL_PATH = (
    "models/screw_detector.pth"
)

IMAGE_PATH = "test.jpg"

NUM_CLASSES = 2

CONFIDENCE_THRESHOLD = 0.99

print(
    "Loading trained model..."
)


model = fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None
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

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)


model.eval()

image = cv2.imread(
    IMAGE_PATH
)


if image is None:

    print(
        "ERROR: Could not load test image."
    )

    exit()


# Keep original
result = image.copy()


rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

tensor = F.to_tensor(
    rgb
)

print(
    "Detecting objects..."
)


with torch.no_grad():

    prediction = model(
        [tensor]
    )[0]


count = 0


for box, label, score in zip(
    prediction["boxes"],
    prediction["labels"],
    prediction["scores"]
):

    confidence = score.item()


    if confidence < CONFIDENCE_THRESHOLD:
        continue


    # Only detect class 1 = Screw
    if label.item() != 1:
        continue

    x1, y1, x2, y2 = (
        box.int().tolist()
    )


    count += 1

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    confidence_percent = (
        confidence * 100
    )


    text = (
        f"Screw {count}: "
        f"{confidence_percent:.1f}%"
    )

    cv2.putText(
        result,
        text,
        (
            x1,
            max(y1 - 10, 25)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

print(
    "OBJECTS DETECTED:",
    count
)

cv2.putText(
    result,
    f"Total Screws: {count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 255),
    2
)

max_width = 1200
max_height = 700

height, width = (
    result.shape[:2]
)


scale = min(
    max_width / width,
    max_height / height,
    1.0
)


new_width = int(
    width * scale
)

new_height = int(
    height * scale
)


display = cv2.resize(
    result,
    (
        new_width,
        new_height
    ),
    interpolation=cv2.INTER_AREA
)

cv2.namedWindow(
    "Custom Screw Detection",
    cv2.WINDOW_NORMAL
)


cv2.imshow(
    "Custom Screw Detection",
    display
)

cv2.imwrite(
    "detected_screws.jpg",
    result
)


print()
print(
    "Result saved as:"
)

print(
    "detected_screws.jpg"
)


print()
print(
    "Press any key to close."
)


cv2.waitKey(0)

cv2.destroyAllWindows()