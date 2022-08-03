from app.custom import Orient
from app.imager import ImageLoader, DefectViewer


if __name__ == '__main__':
    n_samples = 2000
    defect = (DefectViewer(row_chop=15, col_chop=15) << (ImageLoader(defect_class=None) << n_samples))
    defect.category = 'All'

    _ = Orient(do_debug=True, num_jobs=20) << defect
