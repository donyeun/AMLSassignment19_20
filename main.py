import os
import sys

from skimage import io
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import yaml
from A1.task_a1 import TaskA1
from A2.task_a2 import TaskA2
from B1.task_b1 import TaskB1
from B2.task_b2 import TaskB2
from helper import Classifier, DataAnalyst, DataProcessor

cfg = yaml.safe_load(open("config.yaml"))
da = DataAnalyst()
dp = DataProcessor()
cl = Classifier()

# ======================================================================================================================
# Task A1: Gender Detection
sys.stdout = open(cfg['task_a']['a1']['log_path'],"w")          # save logging at designated folder

a1 = TaskA1(cfg, dp, cl)
X, Y = a1.feature_extraction()                                  # feature exctraction
X_train, X_test, Y_train, Y_test = a1.train_test_split(X, Y)    # dataset splitting
clf, acc_a1_train, acc_a1_val = a1.train(X_train, Y_train)      # training pipeline
acc_a1_test = a1.test(X_test, Y_test, clf)                      # inference
print(acc_a1_train, acc_a1_val, acc_a1_test)

sys.stdout.close()                                              # close the logging activity
# ======================================================================================================================

# ======================================================================================================================
# Task A2: Smile Detection
sys.stdout = open(cfg['task_a']['a2']['log_path'],"w")          # save logging at designated folder

a2 = TaskA2(cfg, dp, cl)
X, Y = a2.feature_extraction()                                  # feature exctraction
X_train, X_test, Y_train, Y_test = a2.train_test_split(X, Y)    # dataset splitting
clf, acc_a2_train, acc_a2_val = a2.train(X_train, Y_train)      # training pipeline
acc_a2_test = a2.test(X_test, Y_test, clf)                      # inference
print(acc_a2_train, acc_a2_val, acc_a2_test)

sys.stdout.close()                                              # close the logging activity
# ======================================================================================================================

# ======================================================================================================================
# Task B1: Multiclass Face Shape Detection
sys.stdout = open(cfg['task_b']['b1']['log_path'],"w")          # save logging at designated folder

b1 = TaskB1(cfg, dp, cl)
X, Y = b1.feature_extraction()                                  # feature exctraction
X_train, X_test, Y_train, Y_test = b1.train_test_split(X, Y)    # dataset splitting
clf, acc_b1_train, acc_b1_val = b1.train(X_train, Y_train)      # training pipeline
acc_b1_test = b1.test(X_test, Y_test, clf)                      # inference
print(acc_b1_train, acc_b1_val, acc_b1_test)

sys.stdout.close()                                              # close the logging activity
# ======================================================================================================================

# ======================================================================================================================
# Task B2: Multiclass Eye Colour Detection
sys.stdout = open(cfg['task_b']['b2']['log_path'],"w")          # save logging at designated folder

b2 = TaskB2(cfg, dp, cl)
X, Y = b2.feature_extraction()                                  # feature exctraction
X_train, X_test, Y_train, Y_test = b2.train_test_split(X, Y)    # dataset splitting
clf, acc_b2_train, acc_b2_val = b2.train(X_train, Y_train)      # training pipeline
acc_b2_test = b2.test(X_test, Y_test, clf)                      # inference
print(acc_b2_train, acc_b2_val, acc_b2_test)

sys.stdout.close()                                              # close the logging activity
# ======================================================================================================================

# ======================================================================================================================
print('TA1:{},{},{};TA2:{},{},{};TB1:{},{},{};TB2:{},{},{};'.format(acc_a1_train, acc_a1_val, acc_a1_test,
                                                        acc_a2_train, acc_a2_val, acc_a2_test,
                                                        acc_b1_train, acc_b1_val, acc_b1_test,
                                                        acc_b2_train, acc_b2_val, acc_b2_test))
# ======================================================================================================================
