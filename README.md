# Week_4-Task-4
brief intro to YOLO/pretrained object detection models — just running inference on a pretrained model, not training yet

ScrewDetector/
|--dataset/
|  |--annotations/
|    |--screw_001.json
|    |--screw_002.json
|    .
|    .
|     
|    
|  |--images/
|    |--screw_001.jpg
|    |--screw_002.jpg
|    .
|    .
|
|--models/
|  |--screw_detector.pth
|
|--dataset.py
|--train.py
|--detect.py
|--test.jpg

Steps to be followed:
First, open this folder in VS Code or the Command Prompt.
Create a Virtual environment by running:
  1. python -m venv venv
  2. venv\Scripts\activate
  3. Then you will see something like this "(venv) PS D:\YourProject>".
(Note: A virtual environment creates an isolated Python environment for a project, so its packages and versions don't conflict with those of other projects.)
