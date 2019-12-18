import csv
import pickle
from collections import Counter
import dlib
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, feature, io
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm
import os
import cv2
from imutils import face_utils


class DataAnalyst:
    """Some notes

    Attributes:
        name: A string representing the customer's name.
    """

    # def __init__():
    
    #     """some notes"""
    #     self.name = name
    #     self.balance = balance

    def check_pixel_size_in_dataset(self, dataset_foldername, filename_list):
        """Notes
        """
        pixel_size_list = []
        for filename in filename_list:
            filepath = os.path.join(dataset_foldername, filename)
            width, height = Image.open(filepath).size
            pixel_size_list.append(str(width) + ", " + str(height))
        pixel_size_df = pd.DataFrame(pixel_size_list, columns=['img_dimension'])
        print(pixel_size_df['img_dimension'].value_counts())
        return pixel_size_df['img_dimension'].value_counts(), pixel_size_df['img_dimension'].hist()

class DataProcessor:
    """Notes
    """

    def determine_X_and_Y_set_from_label_file(self, label_csv_filepath, x_header_name, y_header_name, delimiter='\t'):
        """Determine the X (the input features) and the Y (label) from a CSV file
        """
        X = []
        Y = []
        with open(label_csv_filepath) as csv_file:
            data = csv.DictReader(csv_file, delimiter=delimiter)
            for row in data:
                X.append(row[x_header_name])
                Y.append(row[y_header_name])
        return X, Y

    def hog_face_detector(self, input_dir, img_filename, face_detector, output_dir):
        """Detect face using HOG
        """
        img_filepath = os.path.join(input_dir, img_filename)
        img = cv2.imread(img_filepath)
        
        rects = face_detector(img, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cropped_img = img[
                y:y+h,
                x:x+w,
            ]
            cv2.imwrite(os.path.join(output_dir, img_filename), cropped_img)

    def flatten_imgs(self, input_dir, X):
        """Take all the images within input_dir and flatten it
        """
        flattened_X = []
        for x in X:
            img = io.imread(os.path.join(input_dir, x))
            flattenned_img = img.flatten()
            flattened_X.append(flattenned_img)
        return flattened_X

    def hog_subregion_detector(self, input_dir, img_filename, face_detector, shape_predictor, shape_point_start, shape_point_end, output_dir):
        """Detect facial features using HOG
        """  
        # https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
        img_filepath = os.path.join(input_dir, img_filename)
        img = cv2.imread(img_filepath)
        
        rects = face_detector(img, 1)
        for (i, rect) in enumerate(rects):
            shape = shape_predictor(img, rect)
            
            x_points = [shape.part(nth_point).x for nth_point in range(shape_point_start, shape_point_end)]
            y_points = [shape.part(nth_point).y for nth_point in range(shape_point_start, shape_point_end)]

            max_x = max(x_points)
            min_x = min(x_points)
            max_y = max(y_points)
            min_y = min(y_points)
            extra = 3

            cropped_img = img[
                    min_y - extra:max_y + extra,
                    min_x - extra:max_x + extra,
            ]

            cv2.imwrite(os.path.join(output_dir, img_filename), cropped_img)

    def crop_subregion_from_dataset(self, filenames, dataset_dir, shape_predictor_dir, shape_point_marks, cropped_dataset_dir):
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(shape_predictor_dir)
        for filename in tqdm(filenames):
            self.hog_subregion_detector(
                dataset_dir,
                filename,
                face_detector,
                shape_predictor,
                shape_point_marks['start'],
                shape_point_marks['end'],
                cropped_dataset_dir
            )

    def crop_face_from_dataset(self, filenames, dataset_dir, cropped_dataset_dir):
        face_detector = dlib.get_frontal_face_detector()
        for filename in tqdm(filenames):
            self.hog_face_detector(
                dataset_dir,
                filename,
                face_detector,
                cropped_dataset_dir
            )

    def raw_imgs_to_lbp_hists(self, image_dir, image_filenames, lbp_params):
        """Extract LBP histograms from raw images 
        """
        hists = []
        for filename in tqdm(image_filenames):
            filepath = os.path.join(image_dir, filename)

            # load the image file
            image = io.imread(filepath)

            # convert RGB image into greyscale
            grey_image = color.rgb2grey(image)

            # convert greyscale image into lbp features
            lbp_image = feature.local_binary_pattern(
                grey_image,
                lbp_params['n_points'],
                lbp_params['radius'],
                lbp_params['method'],
            )

            # convert lbp features into histogram
            n_bins = int(lbp_image.max() + 1)
            hist, _ = np.histogram(
                lbp_image.ravel(),
                bins=n_bins,
                range=(0, n_bins)
            )
            hists.append(hist)
        return hists

    def haar_cascade_cropping(self, input_dir, img_filename, haar_cascade_filepath, scale_factor, min_neighbors, output_dir):
        """Detect object that matters using Haar Cascade, crop the image,
           then save the cropped image to a local folder
        """
        img_filepath = os.path.join(input_dir, img_filename)
        haar_cascade = cv2.CascadeClassifier(haar_cascade_filepath)
        img = cv2.imread(img_filepath)
    #     $$$# https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters
        obj = haar_cascade.detectMultiScale(img, scale_factor, min_neighbors)
        # reduce_px = 10
        for (x, y, w, h) in obj:
            cropped_img = img[y+int(h/3):y+h-int(h/3), x+int(w/3):x+w-int(w/3)]
            cropped_img = cv2.resize(cropped_img, dsize=(25, 25))
            cv2.imwrite(os.path.join(output_dir, img_filename), cropped_img)

    def crop_subregion_from_dataset_with_haar(self, filenames, dataset_dir, task_cfg, cropped_dataset_dir):
        # face_detector = dlib.get_frontal_face_detector()
        # shape_predictor = dlib.shape_predictor(shape_predictor_dir)
        for filename in tqdm(filenames):
            self.haar_cascade_cropping(
                dataset_dir,
                filename,
                task_cfg['haar_cascade_path'],
                1.3,
                3,
                cropped_dataset_dir
            )

class Classifier:
    def LinearSVM(self, X, Y, task_cfg, train_cfg):
        # SVM classifier with hyperparam tuning
        try:
            clf = pickle.load(
                open(task_cfg['model_path'], "rb")
            )
            print("local model is found and being used instead of retraining the model")
        except (OSError, IOError):
            print("training SVM (and save the resulted model afterwards to a local pickle file)")
            # Do grid search and k-cross validation to SVM Classifier
            pipeline = make_pipeline(
                StandardScaler(),
                LinearSVC(
                    random_state = train_cfg['random_state'],
                    tol = train_cfg['tol'],
                    max_iter = train_cfg['max_iter'],
                ),
            )
            param_candidates = {
                'linearsvc__C' : task_cfg['svm']['param_candidates']['C'],
            }

            clf = GridSearchCV(
                pipeline,
                param_candidates,
                cv = train_cfg['k_crossval'],
                return_train_score = True,
                n_jobs = -1,
                verbose = 51,
                )


            # clf = make_pipeline(
            #     StandardScaler(),
            #     GridSearchCV(
            #         LinearSVC(
            #             random_state = train_cfg['random_state'],
            #             tol = train_cfg['tol'],
            #             max_iter = train_cfg['max_iter'],
            #         ),
            #         task_cfg['svm']['param_candidates'],
            #         cv = train_cfg['k_crossval'],
            #         refit = True
            #         # iid = False,
            #         n_jobs = -1,
            #         verbose = 51,
            #     ),
            # )
            clf.fit(X, Y)

            # save all the models resulted to a local pickle
            pickle.dump(
                clf,
                open(task_cfg['model_path'], "wb")
            )

        print(clf)
        print("Best Estimator: \n{}\n".format(clf.best_estimator_))
        print("Best Parameters: \n{}\n".format(clf.best_params_))
        print("Best Test Score: \n{}\n".format(clf.best_score_))
        print("Best Training Score: \n{}\n".format(clf.cv_results_['mean_train_score'][clf.best_index_]))
        print("All Training Scores: \n{}\n".format(clf.cv_results_['mean_train_score']))
        print("All Test Scores: \n{}\n".format(clf.cv_results_['mean_test_score']))

        return clf