import os
import sys
import traceback
sys.path.append(os.path.join(os.path.abspath(""), ".."))
from sklearn.linear_model import LogisticRegression
from app.models import Classifier
from app.evolutionary import Evolver
from app.imager import ImageLoader, DefectViewer
from app.custom import Orient, HighlightFrontGrid


# This is the data processing and model we are trying to optimize
class OptimizeModel:
    def __init__(self, defect, clean):
        self.defect = defect
        self.clean = clean
        # Setting up default parameters which will then be overridden by the get function
        self.default_params = {'penalty': 'l2', 'seed': 14376, 'pca_dims': None, 'num_jobs': 20, 'reduce_max': 1,
                               'finger_mult': 1}

    def get(self, **kwargs):
        """
        Optimizer assumes there is a get function  and calls it with the list of parameters
        """

        # noinspection PyBroadException
        try:
            # The default dictionary is updated with input values
            self.default_params.update(kwargs)

            # Call the Classifier with inputs:
            # 1. Image wrapper with defects
            # 2. Image wrapper with clean images
            # 3. The model class
            # 4. The data manipulation class
            cla = Classifier(self.defect.copy(), self.clean.copy(), LogisticRegression, HighlightFrontGrid)

            # When done, return the score
            score = -cla.fit(**self.default_params)
            return score
        except Exception:
            #  Print any expcetions and return it as a string
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            strng = 'Exception: '.join('!! ' + line for line in lines)
            print(strng)
            return strng


if __name__ == '__main__':
    n_samples = 400
    defect = (DefectViewer(row_chop=25, col_chop=25) << (
                ImageLoader(defect_class='FrontGridInterruption') << n_samples))
    defect.category = 'FrontGridInterruption'
    clean = (DefectViewer(row_chop=25, col_chop=25) << (ImageLoader(defect_class='None') << n_samples))
    clean.category = 'None'

    # Get the oriented images and HOG
    oriented_defect = Orient(num_jobs=20) << defect
    oriented_clean = Orient(num_jobs=20) << clean

    # Create an evolver class with the parameters to optimize
    # Idelally this shoudl be parallelizable but for whatever reason it hangs. Set num_jobs to 1 for that reason
    evo = Evolver(OptimizeModel, {'defect': oriented_defect[-1], 'clean': oriented_clean[-1]}, num_jobs=1)

    # The format for adding features is
    # name, minimum value, maximum value, data type (int or float)
    evo.add_feature('num_jobs', 20, 20, int)
    evo.add_feature('reduce_max', 0, 1, int)
    evo.add_feature('padding_mult', 3, 20, int)
    evo.add_feature('finger_mult', 0, 100, float)
    evo.add_feature('finger_height', 3, 50, int)
    evo.add_feature('max_finger_width', 3, 9, int)
    evo.add_feature('pca_dims', 1, n_samples - 1, int)

    results_df = evo.run(num_samples=30, generations=40)
    print(results_df)
    results_df.to_csv('evolver_results.csv')
