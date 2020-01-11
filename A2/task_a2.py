import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TaskA2:
    def __init__(self, configurationYamlObject, dataProcessorObject, classifierObject):
        self.cfg = configurationYamlObject
        self.dp = dataProcessorObject
        self.cl = classifierObject

    def feature_extraction(self, test_without_retrain=False):
        """This feature extraction function has several steps:
           1. load the X (image filenames) and Y (class label),
           2. crop the region of interest (mouth in this case) from the image,
           3. determine the final X and Y (because some images will be lost from step #2
              due to failure in detecting area of interest), and
           4. execute the feature extraction: raw image -> greyscale -> LBP -> histogram
        
           Return:
            X_final: the LBP histogram of X train
            Y_final: the list of Y (class label) after feature extraction process
        """
        print('Feature Engineering')
        if not test_without_retrain:
            # to accomodate additional test-set
            dataset_param = self.cfg['task_a']['general']
        else:
            dataset_param = self.cfg['task_a']['additional_test_set']
            self.dp.emptying_folder(self.cfg['task_a']['a2']['cropped_dataset_dir'])

        # load the X and Y label from CSV
        print('Load the X and Y label from CSV')
        X, Y = self.dp.determine_X_and_Y_set_from_label_file(
            dataset_param['label_csv_path'],
            dataset_param['x_header_name'],
            self.cfg['task_a']['a2']['y_header_name']
        )

        # crop the mouth region from the image.
        # the region of interest within some images might not be detected,
        # so the cropped images dataset might be smaller in quantity than the original dataset
        print('Crop the mouth region from the image')
        cropped_dataset_dir = self.cfg['task_a']['a2']['cropped_dataset_dir']
        self.dp.crop_subregion_from_dataset(
            X,
            dataset_param['dataset_dir'],
            self.cfg['shape_predictor']['model_dir'],
            self.cfg['shape_predictor']['mouth'],
            cropped_dataset_dir
        )

        # determine the final X and Y after cropping
        print("Determine the final X and Y after cropping")
        X_final = []
        Y_final = []
        for ori_img_filename in X:
            if ori_img_filename in os.listdir(cropped_dataset_dir):
                X_final.append(ori_img_filename)
                row_num = X.index(ori_img_filename)
                Y_final.append(Y[row_num])

        # feature engineering (raw image -> greyscale -> LBP -> histogram)
        print("Feature Engineering (raw image->greyscale->LBP->histogram)")
        X_final = self.dp.raw_imgs_to_lbp_hists(
            cropped_dataset_dir,
            X_final,
            self.cfg['task_a']['a2']['lbp'],
        )
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
            self.cfg['task_a']['a2'],
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
