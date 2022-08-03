from app.models import Classifier
from app.models import VectorClassifier
from app.imager import ImageLoader, DefectViewer


if __name__ == '__main__':

    n_samples = 50
    filename_df = ImageLoader(defect_class=None).get()
    filename_df = DefectViewer().get(filename_df)

