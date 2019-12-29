import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TaskB1:
    def __init__(self, configurationYamlObject, dataProcessorObject, classifierObject):
        self.cfg = configurationYamlObject
        self.dp = dataProcessorObject
        self.cl = classifierObject

    def feature_extraction(self):
        """This feature extraction function has several steps:
           1. load the X (image filenames) and Y (class label),
           2. crop the region of interest (eye in this case) from the image,
           3. determine the final X and Y (because some images will be lost from step #2
              due to failure in detecting area of interest), and
           4. open the image and flatten
        
           Return:
            X_final: the LBP histogram of X train
            Y_final: the list of Y (class label) after feature extraction process
        """
        print('Feature Engineering')

        # load the X and Y label from CSV
        print('Load the X and Y label from CSV')
        X, Y = self.dp.determine_X_and_Y_set_from_label_file(
            self.cfg['task_b']['general']['label_csv_path'],
            self.cfg['task_b']['general']['x_header_name'],
            self.cfg['task_b']['b1']['y_header_name']
        )

        # crop the face shape region from the image.
        # the region of interest within some images might not be detected,
        # so the cropped images dataset might be smaller in quantity than the original dataset
        print('Crop the face shape region from the image')
        cropped_dataset_dir = self.cfg['task_b']['b1']['cropped_dataset_dir']
        self.dp.detect_face_shape_from_dataset(
            X,
            self.cfg['task_b']['general']['dataset_dir'],
            self.cfg['shape_predictor']['model_dir'],
            self.cfg['shape_predictor']['jaw'],
            cropped_dataset_dir
        )

        # determine the final X and Y after cropping
        print("Determine the final X and Y after cropping")
        X_csv = []
        Y_final = []
        for ori_img_filename in X:
            stripped_filename = ori_img_filename.strip('.png')
            csv_filename = stripped_filename + '.csv'
            if csv_filename in os.listdir(cropped_dataset_dir):
                X_csv.append(ori_img_filename)
                row_num = X.index(ori_img_filename)
                Y_final.append(Y[row_num])

        # open the image and flatten it
        print("Open the image and flatten it")
        X_final = []
        for x in X_csv:
            stripped_filename = x.strip('.png')
            csv_filename = stripped_filename + '.csv'
            with open(os.path.join(cropped_dataset_dir, csv_filename), 'r') as txt_file:
                lines = txt_file.readlines()
                x_final = []
                for line in lines:
                    x_coord, y_coord = line.strip().split('\t')
                    x_final.append(
                        [x_coord, y_coord]
                    )
            X_final.append(x_final)
        X_final = np.asarray(X_final)
        X_final = X_final.reshape(X_final.shape[0], -1)
        return X_final, Y_final

    def train_test_split(self, X, Y):
        """This feature extraction function has several steps:
           1. load the X (image filenames) and Y (class label)
           2. Crop the region of interest (mouth in this case) from the image
           3. determine the final X and Y (because some images will be lost from step #2
              due to failure in detecting area of interest.)
        
           Return:
            X_train: the LBP histogram of X train
            Y_cropped: the list of Y (class label) after feature extraction process
        """
        print("Training and Testing Dataset Splitting")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size = self.cfg['train']['test_proportion'],
            random_state = self.cfg['train']['random_state'],
        )
        return X_train, X_test, Y_train, Y_test

    def train(self, X_train, Y_train):
        """Train an SVM linear kernel classifier, along with
           cross-validation and hyperparameter tuning techniques.

           Return:
            clf: the resulted model
            acc_train: best accuracy score of the training data
            acc_val: best accuracy score of the validation data
        """
        print("Training with Linear SVM Classifier")
        # classification
        clf = self.cl.LinearSVM(
            X_train,
            Y_train,
            self.cfg['task_b']['b1'],
            self.cfg['train'],
        )

        acc_train = clf.cv_results_['mean_train_score'][clf.best_index_]
        acc_val = clf.best_score_
        return clf, acc_train, acc_val

    def test(self, X_test, Y_test, clf):
        """Test the resulted model with testing dataset

           Return:
            accuracy score of the training dataset
        """
        print("Testing the classifier with testing dataset")
        return accuracy_score(
            Y_test,
            clf.predict(
                X_test)
            )
