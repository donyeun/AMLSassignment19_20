import pandas as pd
import matplotlib.pyplot as plt
# TASK A
TASK_A_DATASET_DIR = 'Datasets/celeba/img/'
TASK_A_LABEL_CSV_FILEPATH = 'Datasets/celeba/labels.csv'
TASK_A_X_HEADER_NAME = 'img_name'
TASK_A1_Y_HEADER_NAME = 'gender'
TASK_A2_Y_HEADER_NAME = 'smiling'

# TASK B
TASK_B_DATASET_DIR = 'Datasets/cartoon_set/img/'
TASK_B_LABEL_CSV_FILEPATH = 'Datasets/cartoon_set/labels.csv'
TASK_B_X_HEADER_NAME = 'file_name'
TASK_B1_Y_HEADER_NAME = 'face_shape'
TASK_B2_Y_HEADER_NAME = 'eye_color'

fig, axes = plt.subplots(nrows=2, ncols=2)
CSV_LABEL_SEPARATOR_MARK = '\t'


# Task A1 dataset            
task_a_label_df = pd.read_csv(
                    TASK_A_LABEL_CSV_FILEPATH,
                    sep=CSV_LABEL_SEPARATOR_MARK
                )
task_a_label_df[TASK_A1_Y_HEADER_NAME] = task_a_label_df[TASK_A1_Y_HEADER_NAME].apply(str)
bar = task_a_label_df[TASK_A1_Y_HEADER_NAME].value_counts().plot(kind='bar', title='Task A1: ' + TASK_A1_Y_HEADER_NAME , ax=axes[0,0])
bar.set_xlabel('class label')
bar.set_ylabel('quantity')

# Task A2 dataset            
task_a_label_df = pd.read_csv(
                    TASK_A_LABEL_CSV_FILEPATH,
                    sep=CSV_LABEL_SEPARATOR_MARK
                )
task_a_label_df[TASK_A2_Y_HEADER_NAME] = task_a_label_df[TASK_A2_Y_HEADER_NAME].apply(str)
bar = task_a_label_df[TASK_A2_Y_HEADER_NAME].value_counts().plot(kind='bar', title='Task A2: ' + TASK_A2_Y_HEADER_NAME , ax=axes[0,1])
bar.set_xlabel('class label')
bar.set_ylabel('quantity')

# Task B1 dataset
task_b_label_df = pd.read_csv(
                    TASK_B_LABEL_CSV_FILEPATH,
                    sep=CSV_LABEL_SEPARATOR_MARK
                )
task_b_label_df[TASK_B1_Y_HEADER_NAME] = task_b_label_df[TASK_B1_Y_HEADER_NAME].apply(str)
bar = task_b_label_df[TASK_B1_Y_HEADER_NAME].value_counts().plot(kind='bar', title='Task B1: ' + TASK_B1_Y_HEADER_NAME, ax=axes[1,0])
bar.set_xlabel('class label')
bar.set_ylabel('quantity')

# Task B2 dataset
task_b_label_df = pd.read_csv(
                    TASK_B_LABEL_CSV_FILEPATH,
                    sep=CSV_LABEL_SEPARATOR_MARK
                )
task_b_label_df[TASK_B2_Y_HEADER_NAME] = task_b_label_df[TASK_B2_Y_HEADER_NAME].apply(str)
bar = task_b_label_df[TASK_B2_Y_HEADER_NAME].value_counts().plot(kind='bar', title='Task B2: ' + TASK_B2_Y_HEADER_NAME, ax=axes[1,1])
bar.set_xlabel('class label')
bar.set_ylabel('quantity')

plt.show()