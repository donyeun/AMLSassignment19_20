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
    right_eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')
    img = cv2.imread(filepath)
    right_eye = right_eye_cascade.detectMultiScale(img, 1.7, 11)
    for (x, y, w, h) in right_eye:
        cropped_right_eye = img[y:y+h, x:x+w]
        # print(cropped_right_eye)
        # print(cropped_right_eye)
        cv2.imwrite('mouth/' + filename, cropped_right_eye)


celeba_dataset_foldername = 'Datasets/celeba/img/'
for filename in tqdm(os.listdir(celeba_dataset_foldername)):
    if 'jpg' not in filename:
        continue
    else:
        right_eye_detection(celeba_dataset_foldername, filename)