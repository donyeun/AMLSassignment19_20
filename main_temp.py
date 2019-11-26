import csv
import os
import cv2
import numpy as np
from skimage import color, feature, io
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def determine_X_and_Y_set_from_label_file(label_csv_fileloc, x_header_name, y_header_name, delimiter='\t'):
    X = []
    Y = []
    with open(label_csv_fileloc) as csv_file:
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




label_csv_fileloc = 'Datasets/celeba/labels.csv'
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

X, Y = determine_X_and_Y_set_from_label_file(label_csv_fileloc, x_header_name, y_header_name)


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20
)

# X_train = X_train[:100]
# X_test = X_test[:100]
# Y_train = Y_train[:100]
# Y_test = Y_test[:100]

X_lbp_images = []
# X_lbp_images_raw = [] # REMOVE IT
for x_train_filename in tqdm(X_train):
    filename = os.path.join(celeba_dataset_foldername, x_train_filename)
    image = io.imread(filename)
    # grey_image = color.rgb2gray(image)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # grey_image = data.brick()
    lbp_image = feature.local_binary_pattern(
        grey_image,
        # image,
        16,
        2,
        method="uniform"
    )
    # print(lbp_image)
    n_bins = int(lbp_image.max() + 1)
    # n_bins = 2      # why is it we only have two bins when n_bins = int(lbp_image.max() + 1)????? This sort of explains the 0.5 situasion
    # print(lbp_image)
    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    # print("before ", hist.shape)
    # hist = hist.reshape(1, 10)
    # print("after ", hist.shape)
    # print(hist)
    # print("should be ", hist.shape)
    X_lbp_images.append(hist)
    # X_lbp_images_raw.append(image.reshape(1, -1)) #RAW IMAGE

# X_lbp_images = np.array(X_lbp_images)

# expand dataset


# print(X_lbp_images.shape)
# nsamples, nx, ny = X_lbp_images.shape
# d2_train_dataset = X_lbp_images.reshape((nsamples,nx*ny))
# nsamples, nx = X_lbp_images.shape
# d2_train_dataset = X_lbp_images.reshape(nsamples, nx)
# print(d2_train_dataset.shape)
clf = svm.SVC(kernel='linear', verbose=True)

# clf = svm.SVC(kernel='rbf')
clf.fit(X_lbp_images, Y_train)
# clf.fit(X_lbp_images_raw, Y_train)

Y_pred = []
X_test_lbp_images = []
for x_test_filename in tqdm(X_test):
    filename = os.path.join(celeba_dataset_foldername, x_test_filename)
    image = io.imread(filename)
    # grey_image = color.rgb2gray(image)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = feature.local_binary_pattern(
        grey_image,
        16,
        2,
        method="uniform"
    )
    # lbp_image = np.array(lbp_image)
    n_bins = int(lbp_image.max() + 1)
    # print(n_bins)
    # n_bins = 2
    # I previously removed the "density=True parameter"
    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    # print('~~~~',hist.shape)
    # nx = hist.shape
    # print("before ", hist.shape)
    # hist = hist.reshape(1, 10)
    # 
    # hist = hist.reshape(-1, 1)
    # print("after ", hist.shape) 
    X_test_lbp_images.append(hist)
# d2_train_dataset = X_lbp_images.reshape(nsamples, nx)
    # nx, ny = lbp_image.shape
    # lbp_image = lbp_image.reshape(1, nx*ny)
    # Y_pred.append(clf.predict(lbp_image))
    # print(clf.predict(hist)[0])

    # Y_pred.append(clf.predict(image))  #RAW IMAGE
Y_pred = clf.predict(X_test_lbp_images)
print(Y_test[:20])
print(Y_pred[:20])
print(accuracy_score(Y_test, Y_pred))
# plt.show()