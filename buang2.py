import csv
import os
import cv2
import numpy as np
from skimage import color, feature, io
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def right_eye_detection(foldername, filename):
    filepath = os.path.join(foldername, filename)
    right_eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    img = cv2.imread(filepath)
    # print(img)
    right_eye = right_eye_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in right_eye:
        cropped_right_eye = img[y:y+h, x:x+w]
        # print(cropped_right_eye)
        cv2.imwrite('right_eye/' + filename, cropped_right_eye)


cartoon_set_dataset_foldername = 'Datasets/cartoon_set/img/'

for filename in tqdm(os.listdir(cartoon_set_dataset_foldername)):
    if 'png' not in filename:
        continue
    else:
        right_eye_detection(cartoon_set_dataset_foldername, filename)