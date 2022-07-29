import os
import sys
sys.path.append(os.path.join(os.path.abspath(""), ".."))
from app.local_descriptors import KAZE
from app.imager import ImageLoader, DefectViewer, Show


if __name__ == '__main__':
    # Load 10 examples and name the category for it. Category is like a title for images
    n_samples = 40
    defect = (DefectViewer(row_chop=25, col_chop=25) << (
            ImageLoader(defect_class='FrontGridInterruption') << n_samples))
    defect.category = 'FrontGridInterruption'

    clean = (DefectViewer(row_chop=25, col_chop=25) << (
             ImageLoader(defect_class='None') << n_samples))
    clean.category = 'None'

    #
    kaze = KAZE() << defect
    Show(num_images=5) << kaze

    kaze = KAZE()
    t_histos = kaze.fit(~defect)
    histos = kaze.get(~clean)

    #
    pass
