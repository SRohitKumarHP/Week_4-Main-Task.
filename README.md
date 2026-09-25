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

After creation of Virtual Environment make sure that the dataset is in the exact format as given
  dataset/
  |--annotation/
  |  |--screw_001.json
  |  |--screw_002.json
  |  .
  |  .
  |--images/
  |  |--screw_001.jpg
  |  |--screw_002.jpg
  |  .
  |  .

The dataset is annotated and generated using GPT. You can also use any annotated dataset that is ready to use. But it should have 'annotations' and an 'images' folder as above.

After the Folder and dataset is ready. 

First run bash: (venv) PS D:\YourProject> python dataset.py

Next run bash: (venv) PS D:\YourProject> python train.py {Training may take some time so keep the dataset short and simple if you use any of your own dataset}

After training is complete run bash: (venv) PS D:\YourProject> python detect.py.

At the end, you will get the detected_screws.jpg
