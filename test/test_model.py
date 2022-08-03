import os
import sys
import tabulate
from sklearn.ensemble import GradientBoostingClassifier
sys.path.append(os.path.join(os.path.abspath(""), ".."))
from app.models import Classifier
from app.imager import ImageLoader, DefectViewer, Exposure


if __name__ == '__main__':
    # Load 10 examples and name the category for it. Category is like a title for images
    n_samples = 400
    defect_class = 'FrontGridInterruption'
    defect = (DefectViewer(row_chop=25, col_chop=25) << (
                ImageLoader(defect_class='FrontGridInterruption') << n_samples))
    defect.category = 'FrontGridInterruption'
    clean = (DefectViewer(row_chop=25, col_chop=25) << (ImageLoader(defect_class='FrontGridInterruption', is_not=True)
                                                        << n_samples))
    clean.category = 'None'

    params = {'seed': 14376, 'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.1, 'pca_dims': 160,
              'mode': 'gamma', 'gamma': 0.5, 'gain': 2}

    cla = Classifier(defect, clean, GradientBoostingClassifier, Exposure)

    # When done, return the score
    score = cla.fit(**params)
    print(score)

    # Misclassified
    conf, out = cla.misclassified()
    print(tabulate.tabulate([['True 0', conf[0, 0], conf[0, 1]], ['True 1', conf[1, 0], conf[1, 1]]],
                            headers=['', 'Pred 0', 'Pred 1']))
