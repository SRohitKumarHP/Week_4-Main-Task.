# Week_4-Task-4
brief intro to YOLO/pretrained object detection models — just running inference on a pretrained model, not training yet

Follow the document structure below:
```bash
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
```
## Steps to be followed:
First, open this folder in VS Code or the Command Prompt.

## Prerequisites

Make sure Python is installed on your system.

Download or clone this repository and keep all the Python files along with the required `.jpg`, `.png`, or other input image files in the appropriate folder.

## Installation

Before running the programs, install the required Python libraries.

Create a Virtual environment by running:
  1. python -m venv venv
  2. venv\Scripts\activate
  3. Then you will see something like this "(venv) PS D:\YourProject>".
## Note:
(A virtual environment creates an isolated Python environment for a project, so its packages and versions don't conflict with those of other projects.)

Open a new terminal in the project folder and run:

```bash
pip install opencv-python
pip install numpy
```
Extract the dataset.zip in the folder where all the files are available.
After creating the Virtual Environment, make sure the dataset is in the exact format provided after extracting 'dataset.zip'.
```bash
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
```

- The dataset is annotated and generated using GPT. You can also use any annotated dataset that is ready to use. But it should have 'annotations' and an 'images' folder as above.
  
- After the Folder and dataset is ready. 

1. First run bash:
```
(venv) PS D:\YourProject> python dataset.py {This program will evaluate the data set and check whether all the required format is being followed in the dataset folder}
```

2. Next run bash: 
```
(venv) PS D:\YourProject> python train.py {Training may take some time so keep the dataset short and simple if you use any of your own dataset}
```

3. After training is complete run bash: 
```
(venv) PS D:\YourProject> python detect.py.
```

4. At the end, you will get the detected_screws.jpg
