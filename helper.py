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
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

class DataAnalyst:
    """ Data analyst class contains data analysis functions 
    """


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
    """ Data processor class contains image preprocessing and general data processing functions.
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
            if cropped_img.size != 0:
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

    def detect_face_shape_from_dataset(self, filenames, dataset_dir, shape_predictor_dir, shape_point_marks, cropped_dataset_dir):
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(shape_predictor_dir)
        for filename in tqdm(filenames):
            self.hog_face_shape_detector(
                dataset_dir,
                filename,
                face_detector,
                shape_predictor,
                shape_point_marks['start'],
                shape_point_marks['end'],
                cropped_dataset_dir
            )

    def hog_face_shape_detector(self, input_dir, img_filename, face_detector, shape_predictor, shape_point_start, shape_point_end, output_dir):
        """To detect face shape using 68 point face shape predictor
        """
        img_filepath = os.path.join(input_dir, img_filename)
        img = cv2.imread(img_filepath)
        
        rects = face_detector(img, 1)
        for (i, rect) in enumerate(rects):
            shape = shape_predictor(img, rect)
            shape = face_utils.shape_to_np(shape)
            points = shape[shape_point_start:shape_point_end]
            with open(os.path.join(output_dir, img_filename.strip('.png') + '.csv'), 'w') as txt_file:
                for x,y in points:
                    txt_file.write(str(x) + '\t' + str(y) + '\n')
            txt_file.close()

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
        obj = haar_cascade.detectMultiScale(img, scale_factor, min_neighbors)
        for (x, y, w, h) in obj:
            cropped_img = img[y+int(h/3):y+h-int(h/3), x+int(w/3):x+w-int(w/3)]
            cropped_img = cv2.resize(cropped_img, dsize=(25, 25))
            cv2.imwrite(os.path.join(output_dir, img_filename), cropped_img)

    def crop_subregion_from_dataset_with_haar(self, filenames, dataset_dir, task_cfg, cropped_dataset_dir):
        for filename in tqdm(filenames):
            self.haar_cascade_cropping(
                dataset_dir,
                filename,
                task_cfg['haar_cascade_path'],
                1.3,
                3,
                cropped_dataset_dir
            )

    def emptying_folder(self, folder_name):
        for file in os.scandir(folder_name):
            os.unlink(file.path)

class Classifier:
    def plot_learning_curve(self, estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), random_state=None):
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True, random_state=random_state,)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                            fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt

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
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        title = "Learning Curves (Linear SVM)"
        
        self.plot_learning_curve(clf.best_estimator_, title, X, Y, cv=train_cfg['k_crossval'], n_jobs=-1, random_state=train_cfg['random_state'])
        plt.show()

        return clf