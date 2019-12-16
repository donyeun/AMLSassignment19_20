import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import yaml
from helper import Classifier, DataAnalyst, DataProcessor

cfg = yaml.safe_load(open("config.yaml"))
da = DataAnalyst()
dp = DataProcessor()
cl = Classifier()
# print(type(cfg['training']['test_dataset_size_proportion']))
# # ======================================================================================================================
# # Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
# # ======================================================================================================================
# # Task A1
# model_A1 = A1(args...)                 # Build model object.
# acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# # ======================================================================================================================
# Task A2
# load the X and Y
X, Y = dp.determine_X_and_Y_set_from_label_file(
    cfg['task_a']['general']['label_csv_path'],
    cfg['task_a']['general']['x_header_name'],
    cfg['task_a']['a2']['y_header_name']
)
# crop the mouth region from the image. Some mouths might not be detected in 
cropped_dataset_dir = cfg['task_a']['a2']['cropped_dataset_dir']
# dp.crop_mouth_region_from_dataset(X, cfg['task_a']['general']['dataset_dir'], cfg['task_a']['a2']['shape_predictor_dir'], cropped_dataset_dir)

# divide cropped images into training and testing
X_cropped = []
Y_cropped = []
for ori_img_filename in X:
    if ori_img_filename in os.listdir(cropped_dataset_dir):
        X_cropped.append(ori_img_filename)
        row_num = X.index(ori_img_filename)
        Y_cropped.append(Y[row_num])

X_train_filenames, X_test_filenames, Y_train, Y_test = train_test_split(
    X_cropped, Y_cropped, test_size=cfg['train']['test_proportion'], random_state=cfg['train']['random_state']
)

print("training data : ", len(X_train_filenames))
print("testing data  : ", len(X_test_filenames))

# # feature engineering
X_train_hists = dp.raw_imgs_to_lbp_hists(cropped_dataset_dir, X_train_filenames, cfg['task_a']['a2']['lbp'])

# classification
clf = cl.LinearSVM(X_train_hists, Y_train, cfg['task_a']['a2']['model_path'], cfg)

# inference
X_test_hists = dp.raw_imgs_to_lbp_hists(cropped_dataset_dir, X_test_filenames, cfg['task_a']['a2']['lbp'])
print("acc score ", accuracy_score(Y_test, clf.predict(X_test_hists)))


# model_A2 = A2(args...)
# acc_A2_train = model_A2.train(args...)
# acc_A2_test = model_A2.test(args...)
# Clean up memory/GPU etc...


# # ======================================================================================================================
# # Task B1
# model_B1 = B1(args...)
# acc_B1_train = model_B1.train(args...)
# acc_B1_test = model_B1.test(args...)
# Clean up memory/GPU etc...


# # ======================================================================================================================
# # Task B2
# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
# Clean up memory/GPU etc...


# # ======================================================================================================================
# ## Print out your results with following format:
# print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                         acc_A2_train, acc_A2_test,
#                                                         acc_B1_train, acc_B1_test,
#                                                         acc_B2_train, acc_B2_test))

# # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# # acc_A1_train = 'TBD'
# # acc_A1_test = 'TBD'
