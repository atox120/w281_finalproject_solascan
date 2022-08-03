import os
import sys
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
sys.path.append(os.path.join(os.path.abspath(""), ".."))
from app import model_features
from app.models import Classifier
from app.model_features import get_samples
from app.imager import ImageLoader, DefectViewer


def get_data_handler(defect_classes):
    if 'FrontGridInterruption' in defect_classes:
        data_handler = model_features.grid_interruption
    elif 'Closed' in defect_classes:
        data_handler = model_features.closed
    elif 'Isolated' in defect_classes:
        data_handler = model_features.isolated
    elif 'BrightSpot' in defect_classes or 'Corrosion' in defect_classes:
        data_handler = model_features.generic_return
    elif 'ResistiveCrack' in defect_classes:
        data_handler = model_features.resistive_crack
    else:
        raise KeyError('Unsupported model type')

    return data_handler


def _get_models(model_defect_classes, model_params, num_samples, complimentary):

    model_objects = []
    model_classes = []
    model_data_handlers = []
    # For each defect class, create the DataSet
    for defect_classes in model_defect_classes:
        print(defect_classes)
        model_param = model_params[defect_classes]

        # Get the samples for the model
        if isinstance(defect_classes, tuple):
            classes = list(defect_classes)
        else:
            classes = defect_classes

        # Get the data for modeling
        defect, not_defect = get_samples(classes, num_samples, complimentary=complimentary)

        # Get the data handler
        data_handler = get_data_handler(defect_classes)

        # Get the pre processed data for this
        defect_ = data_handler(defect, num_jobs=20)
        not_defect_ = data_handler(not_defect, num_jobs=20)

        # Show the pre and post processed images
        # _ = Show(num_images=2, seed=seed) << (defect, defect_) + (not_defect, not_defect_)

        # Get the parameter for this classifier
        this_param = copy.deepcopy(model_param)
        model_class = this_param['class']
        del this_param['class']

        # Train the classifier
        cla = Classifier(defect_, not_defect_, model_class, None)
        model = cla.fit(**this_param)

        model_objects.append(model)
        model_classes.append(defect_classes)
        model_data_handlers.append(data_handler)

    return model_objects, model_classes, model_data_handlers


def run():
    complimentary = False
    num_samples = 20

    # Analyzing which defect
    model_defect_classes = [('FrontGridInterruption', 'NearSolderPad'), 'Closed', 'Isolated', 'BrightSpot',
                            'Corrosion', 'Resistive']
    model_params = {('FrontGridInterruption', 'NearSolderPad'):
                    {'class': GradientBoostingClassifier, 'n_estimators': 600, 'max_depth': 4,
                     'learning_rate': 0.05, 'pca_dims': min(250, num_samples)},
                    'Closed': {'class': LogisticRegression, 'penalty': 'l2', 'pca_dims': min(200, num_samples)},
                    'Isolated': {'class': GradientBoostingClassifier, 'n_estimators': 300, 'max_depth': 4,
                                 'learning_rate': 0.1, 'pca_dims': min(160, num_samples)},
                    'BrightSpot': {'class': LogisticRegression, 'penalty': 'l2', 'pca_dims': None},
                    'Corrosion': {'class': LogisticRegression, 'penalty': 'l2', 'pca_dims': None},
                    'Resistive':  {'max_features': 0.1, 'min_samples_split': 8, 'random_state': 32}}

    model_objects, model_classes, model_data_handlers = \
        _get_models(model_defect_classes, model_params, num_samples, complimentary)

    img = ImageLoader(defect_class=None, do_train=False)
    filename_df = img.get(n=50)
    filename_df = DefectViewer(row_chop=15, col_chop=15).get(filename_df)

    from app.models import VectorClassifier
    vc = VectorClassifier(model_objects=model_objects, model_classes=model_classes,
                          model_data_handlers=model_data_handlers, defect_classes=img.defect_classes.tolist())
    vc.test(filename_df)


if __name__ == '__main__':
    run()
