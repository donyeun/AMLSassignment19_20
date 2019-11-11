import csv
import os

import numpy as np
from skimage import color, feature, io
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def determine_X_and_Y_set_from_label_file(label_csv_fileloc, x_header_name, y_header_name, delimiter='\t'):
    X = []
    Y = []
    with open (label_csv_fileloc) as csv_file:
        data = csv.DictReader(csv_file, delimiter=delimiter)
        for row in data:
            X.append(row[x_header_name])
            Y.append(row[y_header_name])
    return X, Y




label_csv_fileloc = 'Datasets/celeba/labels.csv'
x_header_name = 'img_name'
y_header_name = 'gender'
X, Y = determine_X_and_Y_set_from_label_file(label_csv_fileloc, x_header_name, y_header_name)


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20)

# X_train = X_train[:100]
# X_test = X_test[:100]
# Y_train = Y_train[:100]
# Y_test = Y_test[:100]

celeba_dataset_foldername = 'Datasets/celeba/img/'
X_lbp_images = []
for x_train_filename in tqdm(X_train):
    filename = os.path.join(celeba_dataset_foldername, x_train_filename)
    image = io.imread(filename)
    grey_image = color.rgb2gray(image)
    lbp_image = feature.local_binary_pattern(
        grey_image,
        1,
        10
    )
    X_lbp_images.append(lbp_image)

X_lbp_images = np.array(X_lbp_images)

# expand dataset


# print(X_lbp_images.shape)
nsamples, nx, ny = X_lbp_images.shape
d2_train_dataset = X_lbp_images.reshape((nsamples,nx*ny))

clf = svm.SVC(kernel='linear')
clf.fit(d2_train_dataset, Y_train)

Y_pred = []
for x_test_filename in tqdm(X_test):
    filename = os.path.join(celeba_dataset_foldername, x_test_filename)
    image = io.imread(filename)
    grey_image = color.rgb2gray(image)
    lbp_image = feature.local_binary_pattern(
        grey_image,
        1,
        10
    )
    lbp_image = np.array(lbp_image)
    nx, ny = lbp_image.shape
    lbp_image = lbp_image.reshape(1, nx*ny)
    Y_pred.append(clf.predict(lbp_image))

print(accuracy_score(Y_test, Y_pred))