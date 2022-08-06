import os
import sys
sys.path.append(os.path.join(os.path.abspath(""), ".."))
from app.local_descriptors import SIFT
from app.imager import ImageLoader, DefectViewer, Show


if __name__ == '__main__':
    # Load 10 examples and name the category for it. Category is like a title for images
    n_samples = 40
    defect_class = 'FrontGridInterruption'
    defect = (DefectViewer(row_chop=25, col_chop=25) << (
            ImageLoader(defect_class='FrontGridInterruption') << n_samples))
    defect.category = 'FrontGridInterruption'

    clean = (DefectViewer(row_chop=25, col_chop=25) << (
             ImageLoader(defect_class='None') << n_samples))
    clean.category = 'None'

    #
    sift = SIFT() << defect
    Show(num_images=5) << sift

    sift = SIFT()
    t_histos = sift.fit(~defect)
    histos = sift.get(~clean)

    pass
