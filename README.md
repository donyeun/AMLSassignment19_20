# APPLIED MACHINE LEARNING SYSTEM ELEC0132 19/20
In an attempt to understand the underlying concepts behind machine learning systems, here we presented with four tasks in the area of computer vision.

The first two tasks are binary classification problems, covering gender classification (A1) and smile detection (A2). The next tasks are multiclass classification which covers the issues of face shape (B1) and eye colour (B2) classification.

Two distinct algorithms are used to detect the region of interest (ROI), namely facial landmark detector and Haar-like features. The same goes with feature extraction, where various signals are utilised across different tasks, such as the raw pixels, coordinate features, and histogram of the local binary pattern (LBP). These different features become the input of an SVM classifier with linear kernel.

During the inference process, each task gained different performance, ranging from 67\% for B1 to 98\% for task B2.

## How to Run the Code
1. Install the Python3 requirements listed within `requirements.txt` by running the following syntax on terminal
```
pip3 install -r requirements.txt
```

2. Run the `main.py` to execute all the training and inference process across 4 tasks. This however may take hours to complete.
```
python3 main.py
```

3. After step 2 is complete. You can see the following resulted files:
    1. model file (the models generated during training process) within `\models\results\<task_number>.pkl`<br>
       For example: [/models/result/a1.pkl](https://github.com/donyeun/AMLSassignment19_20/blob/master/models/result/a1.pkl)
    2. log txt file within `/<task_number>/log.txt` <br>
       For example: [/A1/log.txt](https://github.com/donyeun/AMLSassignment19_20/blob/master/A1/log.txt)

## Structure of repository
    .
    ├── main.py                   # The main Python file that will call helper.py and other Python files
    ├── helper.py                 # Consists of classifier, image preprocessing and data analysis classes
    ├── data_analysis.py          # A Python file to learn data distribution within dataset
    ├── config.yaml               # All the configuration variables such as train-test dataset ratio, etc
    ├── requirements.txt          # The required Python libraries
    ├── report/                   # Latex and pdf report files
    ├── Datasets/
    │   ├── processed/            # The cropped dataset
    │   ├── celeba/               # Original celeba dataset
    │   └── cartoon_set/          # Original cartoon_set dataset
    ├── models/
    │   ├── helper/               # Consists of helper models such as haar-like features and shape predictor
    │   └── result/               # The resulted models from training process
    ├── A1/
    │   ├── task_a1.py            # Feature engineering and classification of task A1
    │   └── log.txt               # The resulted log file during training and inference process
    ├── A2/
    │   ├── task_a2.py            # Feature engineering and classification of task A2
    │   └── log.txt               # The resulted log file during training and inference process
    ├── B1/
    │   ├── task_b1.py            # Feature engineering and classification of task B1
    │   └── log.txt               # The resulted log file during training and inference process
    ├── B2/
    │   ├── task_b2.py            # Feature engineering and classification of task B2
    └── └── log.txt               # The resulted log file during training and inference process
