import os
import sys

from skimage import io
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import yaml
from A1.task_a1 import TaskA1
from A2.task_a2 import TaskA2
from B2.task_b2 import TaskB2
from helper import Classifier, DataAnalyst, DataProcessor

cfg = yaml.safe_load(open("config.yaml"))
da = DataAnalyst()
dp = DataProcessor()
cl = Classifier()

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
# # Task A1
# sys.stdout = open(cfg['task_a']['a1']['log_path'],"w") # for logging purposes

# a1 = TaskA1(cfg, dp, cl)
# X, Y = a1.feature_extraction()
# X_train, X_test, Y_train, Y_test = a1.train_test_split(X, Y)
# clf, acc_a1_train, acc_a2_val = a1.train(X_train, Y_train)
# acc_a1_test = a1.test(X_test, Y_test, clf)
# print(acc_a1_test)

# sys.stdout.close()
# # ======================================================================================================================

# ======================================================================================================================
# Task A2
# sys.stdout = open(cfg['task_a']['a2']['log_path'],"w") # for logging purposes

# a2 = TaskA2(cfg, dp, cl)
# X, Y = a2.feature_extraction()
# X_train, X_test, Y_train, Y_test = a2.train_test_split(X, Y)
# clf, acc_a2_train, acc_a2_val = a2.train(X_train, Y_train)
# acc_a2_test = a2.test(X_test, Y_test, clf)
# print(acc_a2_test)

# sys.stdout.close()
# ======================================================================================================================

# ======================================================================================================================
# Task B2
sys.stdout = open(cfg['task_b']['b2']['log_path'],"w") # for logging purposes

b2 = TaskB2(cfg, dp, cl)
X, Y = b2.feature_extraction()
X_train, X_test, Y_train, Y_test = b2.train_test_split(X, Y)
clf, acc_b2_train, acc_b2_val = b2.train(X_train, Y_train)
acc_b2_test = b2.test(X_test, Y_test, clf)
print(acc_b2_test)

sys.stdout.close()
# ======================================================================================================================

# # load the X and Y
# X, Y = dp.determine_X_and_Y_set_from_label_file(
#     cfg['task_a']['general']['label_csv_path'],
#     cfg['task_a']['general']['x_header_name'],
#     cfg['task_a']['a2']['y_header_name']
# )
# # crop the mouth region from the image. Some mouths might not be detected in 
# cropped_dataset_dir = cfg['task_a']['a2']['cropped_dataset_dir']
# dp.crop_subregion_from_dataset(X, cfg['task_a']['general']['dataset_dir'], cfg['shape_predictor']['model_dir'], cfg['shape_predictor']['mouth'], cropped_dataset_dir)

# # divide cropped images into training and testing
# X_cropped = []
# Y_cropped = []
# for ori_img_filename in X:
#     if ori_img_filename in os.listdir(cropped_dataset_dir):
#         X_cropped.append(ori_img_filename)
#         row_num = X.index(ori_img_filename)
#         Y_cropped.append(Y[row_num])

# X_train_filenames, X_test_filenames, Y_train, Y_test = train_test_split(
#     X_cropped, Y_cropped, test_size=cfg['train']['test_proportion'], random_state=cfg['train']['random_state']
# )
# print("training data : ", len(X_train_filenames))
# print("testing data  : ", len(X_test_filenames))

# # # feature engineering
# X_train_hists = dp.raw_imgs_to_lbp_hists(cropped_dataset_dir, X_train_filenames, cfg['task_a']['a2']['lbp'])

# # classification
# clf = cl.LinearSVM(X_train_hists, Y_train, cfg['task_a']['a2']['model_path'], cfg)

# # inference
# X_test_hists = dp.raw_imgs_to_lbp_hists(cropped_dataset_dir, X_test_filenames, cfg['task_a']['a2']['lbp'])
# print("acc score ", accuracy_score(Y_test, clf.predict(X_test_hists)))
# # ======================================================================================================================

# ======================================================================================================================
# # Task B2
# # open log file
# sys.stdout=open(cfg['task_b']['b2']['log_path'],"w")

# # load the X and Y
# X, Y = dp.determine_X_and_Y_set_from_label_file(
#     cfg['task_b']['general']['label_csv_path'],
#     cfg['task_b']['general']['x_header_name'],
#     cfg['task_b']['b2']['y_header_name']
# )
# # X = X[:200]
# # Y = Y[:200]
# # crop the mouth region from the image. Some mouths might not be detected in 
# cropped_dataset_dir = cfg['task_b']['b2']['cropped_dataset_dir']
# # dp.crop_subregion_from_dataset(X, cfg['task_b']['general']['dataset_dir'], cfg['shape_predictor']['model_dir'], cfg['shape_predictor']['right_eye'], cropped_dataset_dir)
# for x in tqdm(X):
#     dp.haar_cascade_object_cropping(cfg['task_b']['general']['dataset_dir'], x, 'models/helper/haarcascades/haarcascade_eye.xml', 1.3, 3, cropped_dataset_dir)

# # divide cropped images into training and testing
# X_cropped = []
# Y_cropped = []
# for ori_img_filename in X:
#     if ori_img_filename in os.listdir(cropped_dataset_dir):
#         X_cropped.append(ori_img_filename)
#         row_num = X.index(ori_img_filename)
#         Y_cropped.append(Y[row_num])

# # from PIL import Image
# # img = Image.open('ims('RGB').getcolors()

# X_imgs = []
# for x in X_cropped:
#     # img = Image.open(os.path.join(cropped_dataset_dir, x))
#     img = io.imread(os.path.join(cropped_dataset_dir, x))
#     flatten_img = img.flatten()
#     X_imgs.append(flatten_img)
#     # print(img.getcolors())

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_imgs, Y_cropped, test_size=cfg['train']['test_proportion'], random_state=cfg['train']['random_state']
# )
# print("training data : ", len(X_train))
# print("testing data  : ", len(X_test))

# # # classification
# from sklearn.svm import SVC, LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# clf = LinearSVC()
# # print(len(X_train))
# # print(type(X_train))
# # print(X_train[0].shape)
# # print(type(X_train[0]))
# clf = make_pipeline(
#     StandardScaler(),
#     LinearSVC(max_iter=100000, verbose=100)
# )
# clf.fit(X_train, Y_train)


# # inference
# print("acc score ", accuracy_score(Y_test, clf.predict(X_test)))

# # close the log
# sys.stdout.close()
# ======================================================================================================================



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
