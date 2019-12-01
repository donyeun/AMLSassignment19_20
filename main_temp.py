import csv
import os
from collections import Counter

import cv2
import numpy as np
from PIL import Image
from skimage import color, feature, io
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pandas as pd


def determine_X_and_Y_set_from_label_file(label_csv_fullpath, x_header_name, y_header_name, delimiter='\t'):
    X = []
    Y = []
    with open(label_csv_fullpath) as csv_file:
        data = csv.DictReader(csv_file, delimiter=delimiter)
        for row in data:
            X.append(row[x_header_name])
            Y.append(row[y_header_name])
    return X, Y

def face_detection(foldername, filename):
    filepath = os.path.join(foldername, filename)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(filepath)
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_image, 1.1, 4)
    for (x, y, w, h) in faces:
        cropped_img = img[y:y+h, x:x+w]
        cv2.imwrite('cropped/' + filename, cropped_img)

def right_eye_detection(foldername, filename):
    filepath = os.path.join(foldername, filename)
    right_eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_righteye_2splits.xml')
    img = cv2.imread(filepath)
    for (x, y, w, h) in image:
        cropped_right_eye = img[y:y+h, x:x+w]
        cv2.imwrite('right_eye/' + filename, cropped_right_eye)



label_csv_fullpath = 'Datasets/celeba/labels.csv'
# celeba_dataset_foldername = 'cropped/'
celeba_dataset_foldername = 'Datasets/celeba/img/'
x_header_name = 'img_name'
y_header_name = 'gender'

# # crop face
# for filename in tqdm(os.listdir(celeba_dataset_foldername)):
#     # print(filename)
#     if 'jpg' not in filename:
#         continue
#     else:
#         face_detection(celeba_dataset_foldername, filename)

X, Y = determine_X_and_Y_set_from_label_file(label_csv_fullpath, x_header_name, y_header_name)


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20
)

# X_train = X_train[:100]
# X_test = X_test[:100]
# Y_train = Y_train[:100]
# Y_test = Y_test[:100]

X_lbp_images = []
for x_train_filename in tqdm(X_train):
    filename = os.path.join(celeba_dataset_foldername, x_train_filename)
    image = io.imread(filename)
    grey_image = color.rgb2gray(image)
    # grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp_image = feature.local_binary_pattern(
        grey_image,
        # image,
        8,
        2,
        # method="uniform"
    )
    # print(lbp_image)
    n_bins = int(lbp_image.max() + 1)
    # n_bins = 2      # why is it we only have two bins when n_bins = int(lbp_image.max() + 1)????? This sort of explains the 0.5 situasion
    # print(lbp_image)
    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    X_lbp_images.append(hist)
    # X_lbp_images_raw.append(image.reshape(1, -1)) #RAW IMAGE

# X_lbp_images = np.array(X_lbp_images)

# expand dataset

clf = svm.SVC(kernel='linear')
# clf = svm.SVC(kernel='rbf')
clf.fit(X_lbp_images, Y_train)

Y_pred = []
X_test_lbp_images = []
for x_test_filename in tqdm(X_test):
    filename = os.path.join(celeba_dataset_foldername, x_test_filename)
    image = io.imread(filename)
    grey_image = color.rgb2gray(image)
    # grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = feature.local_binary_pattern(
        grey_image,
        8,
        2,
        # method="uniform"
    )
    # lbp_image = np.array(lbp_image)
    n_bins = int(lbp_image.max() + 1)
    # print(n_bins)
    # n_bins = 2
    # I previously removed the "density=True parameter"
    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    X_test_lbp_images.append(hist)

Y_pred = clf.predict(X_test_lbp_images)
print("clf score ", clf.score(X_test_lbp_images, Y_test))
print(Y_test[:20])
print(Y_pred[:20])
print("acc score ", accuracy_score(Y_test, Y_pred))
# plt.show()
