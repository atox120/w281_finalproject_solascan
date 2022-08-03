from app.custom import RemoveBusBars
from app.imager import ImageLoader, DefectViewer, Show


if __name__ == '__main__':

    n_samples = 50
    defects = (DefectViewer(row_chop=25, col_chop=25) << (ImageLoader(defect_class=None) << n_samples))
    defects.category = 'All'

    removed = RemoveBusBars(replace_type='zero', num_jobs=20) << (defects, )

    _ = Show(num_images=n_samples) << (defects, removed[-1])
