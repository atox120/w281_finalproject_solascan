import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from app.utils import ImageWrapper


class Classifier:
    def __init__(self, defect, not_defect, model_class, data_class):

        # These are the images to
        self.defect = defect
        self.not_defect = not_defect
        self.model_class = model_class
        self.data_class = data_class
        self.model = None
        self.pca = None

        self.transformation_label = ''
        self.pre_transform = None
        self.post_transform = None
        self.train_data = ()
        self.cv_data = ()
        self.reshape = None
        self.image_labels = []

    def _process_data(self, params):
        """
        Apply the data object to the images
        """

        if self.data_class is not None:
            # First accumulate all the parameters for the data class
            data_params = {}
            keys = list(params.keys())
            for key in keys:
                if key in self.data_class().__dict__:
                    data_params[key] = params[key]
                    del (params[key])

            # Create an object of the data class
            self.data_obj = self.data_class(**data_params)

            # Process the  data through the data object
            defect = self.data_obj << self.defect
            not_defect = self.data_obj << self.not_defect

            defect = defect[-1]
            not_defect = not_defect[-1]

            # The first items is the defect class type, remove it?
            self.transformation_label = defect.category
        else:
            defect = self.defect
            not_defect = self.not_defect

        return defect, not_defect

    def _apply_pca(self, pca_dims, x_train, x_cv):
        """
        Get the desired number of dimensions from the data
        """

        if pca_dims is not None:
            self.pca = PCA(n_components=pca_dims)

            # Transform the train data
            x_train = self.pca.fit_transform(x_train)

            # Transform the CV data
            x_cv = self.pca.transform(x_cv)
        return x_train, x_cv

    def misclassified(self):
        """
        Creates a confusion matrix and an ImageWrapper object of misclassified images

        :return confusion matrix, tuple of misclassified images
        """

        out_array = []
        confusion = np.zeros((2, 2))
        if self.cv_data:
            # noinspection PyTupleAssignmentBalance
            x_cv, y_true, y_pred, cv_indices = self.cv_data

            for yt in np.unique(y_true):
                for yp in np.unique(y_pred):

                    # These are the mismatched predictions
                    bad_pred = np.logical_and(y_true == yt, y_pred == yp)
                    confusion[yt, yp] = np.sum(bad_pred)

                    if yt == yp:
                        continue
                    # bad_pred = np.logical_and(bad_pred, y_true != y_pred)

                    # These are the image lables for the bad predictions
                    image_labels = self.image_labels[cv_indices[bad_pred].astype(int)]

                    # Reformat it into the image
                    x_mis_pre = self.pre_transform[cv_indices[bad_pred].astype(int), :]
                    x_mis_post = self.post_transform[cv_indices[bad_pred].astype(int), :]

                    x_mis_pre = x_mis_pre.reshape(np.sum(bad_pred), *self.reshape)
                    x_mis_post = x_mis_post.reshape(np.sum(bad_pred), *self.reshape)

                    if not x_mis_pre.size == 0:
                        imw_pre = ImageWrapper(x_mis_pre, image_labels=image_labels.tolist(),
                                               category=f'misclassified True: {yt} Pred: {yp}')
                        imw_post = ImageWrapper(
                            x_mis_post, image_labels=image_labels.tolist(),
                            category=self.transformation_label + f'\nmisclassified True: {yt} Pred: {yp}')

                        out_array.append((imw_pre, imw_post))

        return confusion, out_array

    def fit(self, **params):
        """
        1. Apply the params to the data_class and model_class a
        2. Apply PCA
        3. Train the model
        4. Test the model against CV data
        """

        try:
            pca_dims = params['pca_dims']
            del params['pca_dims']
        except KeyError:
            pca_dims = None

        try:
            seed = params['seed']
            del params['seed']
        except KeyError:
            seed = random.randint(0, 2 ** 32)

        # Extract the images before transformation

        # Get the data as numpy array
        pre_defect = ~self.defect
        pre_not_defect = ~self.not_defect

        x_vals = np.vstack((pre_defect, pre_not_defect))
        self.pre_transform = x_vals

        # This is where the Data object class is applied
        defect, not_defect = self._process_data(params)

        # The remaining parameters go into the model class
        self.model = self.model_class(**params)
        self.image_labels = np.array(defect.image_labels + not_defect.image_labels)

        # Get the data as numpy array
        defect = ~defect
        not_defect = ~not_defect

        # These are the values to reshape the images back to
        self.reshape = not_defect.shape[1:]

        # Flatten the image and convert into X array
        defect = defect.reshape((defect.shape[0], -1))
        not_defect = not_defect.reshape((not_defect.shape[0], -1))
        x_vals = np.vstack((defect, not_defect))
        self.post_transform = x_vals

        # Add indices for referencing later
        x_vals = np.hstack((np.arange(x_vals.shape[0])[:, np.newaxis], x_vals))

        # Flatten the image and convert into y array
        y_defect = np.ones((defect.shape[0],)).astype(int)
        y_not_defect = np.zeros((not_defect.shape[0],)).astype(int)
        y_vals = np.concatenate((y_defect, y_not_defect))

        # Split into train and cv set
        x_train, x_cv, y_train, y_cv = \
            train_test_split(x_vals, y_vals, test_size=0.20, random_state=seed)

        # Apply the PCA to the train set and transform the cv set too
        cv_indices = x_cv[:, 0]
        x_train, x_cv = self._apply_pca(pca_dims, x_train[:, 1:], x_cv[:, 1:])

        # Fit the model for the train data
        self.model.fit(x_train, y_train)

        # Predict the model on the cv
        y_pred = self.model.predict(x_cv)

        # Save the CV indices to refernce later
        self.cv_data = (x_cv, y_cv, y_pred, cv_indices)

        # Calculate the balanced accuracy score
        score = balanced_accuracy_score(y_cv, y_pred)

        return score
