## How to Run the Code
1. Install the Python3 requirements listed within `requirements.txt` by running the following syntax on terminal
```
pip3 install -r requirements.txt
```

2. Run the `main.py` to execute all the training and inference process across 4 tasks. This however may take hours to complete.
```
python3 main.py
```

3. After step 2 is complete. You can the following resulted files:
    1. model file (the models generated during training process) within `\models\results\<task_number>.pkl`<br>
       For example: [/models/result/a1.pkl](https://github.com/donyeun/AMLSassignment19_20/blob/master/models/result/a1.pkl)
    2. log txt file within `/<task_number>/log.txt` <br>
       For example: [/A1/log.txt](https://github.com/donyeun/AMLSassignment19_20/blob/master/A1/log.txt)

## Structure of repository
    .
    ├── main.py                   # The main Python file that will call helper.py and other Python files
    ├── helper.py                 # Consists of classifier, image preprocessing and data analysis class
    ├── data_analysis.py          # A Python file to learn data distribution within dataset
    ├── config.yaml               # All the configuration variables such as train-test dataset ratio, etc
    ├── requirements.txt          # The required Python libraries
    ├── Datasets/
    │   ├── processed/            # The cropped dataset
    │   ├── celeba/               # Original celeba dataset
    │   └── cartoon_set/          # Original cartoon_set dataset
    ├── models/
    │   ├── helper/               # Consists of helper models such as haar-like features and shape predictor
    │   └── result/               # The resulted models from training process
    ├── A1/
    │   ├── task_a1.py/           # Feature engineering and classification of task A1
    │   └── log.txt/              # The resulted log file during training and inference process
    ├── A2/
    │   ├── task_a2.py/           # Feature engineering and classification of task A2
    │   └── log.txt/              # The resulted log file during training and inference process
    ├── B1/
    │   ├── task_b1.py/           # Feature engineering and classification of task B1
    │   └── log.txt/              # The resulted log file during training and inference process
    ├── B2/
    │   ├── task_b2.py/           # Feature engineering and classification of task B2
    └── └── log.txt/              # The resulted log file during training and inference process
