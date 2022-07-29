import copy
import time
import math
import numpy as np
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve, convolve1d
from skimage.feature import blob_hog as sk_blob_hog
from skimage.feature import blob_dog as sk_blob_dog
from skimage.feature import blob_log as sk_blob_log


from app.imager import ImageLoader, DefectViewer, Show
from app.utils import input_check, ImageWrapper, line_split_string, parallelize


class BlobDog:
    """
    Finds blobs using the difference of gaussian method. 
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_dog
    """

    def __init__(self, min_sigma, max_sigma, threshold, consume_kwargs=True, **kwargs):
        """
        see https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_dog
        :param min_sigma: The minimum standard deviation for Gaussian kernel. Keep this low to detect smaller blobs. 
        :param max_sigma: The maximum standard deviation for Gaussian kernel. Keep this high to detect larger blobs. 
        :param threshold: The absolute lower bound for scale space maxima. Local maxima smaller than threshold are ignored. Reduce this to detect blobs with lower intensities. 
        :param consume_kwargs: If True check for empty kwargs
        :param **kwargs: see below

        :keyword Arguments:
            :sigma_ratio:
            :overlap:
            :threshold_rel:
            :exclude_border:
        """

        self.params = {'min_sigma': min_sigma, 'max_sigma':max_sigma, 'threshold':threshold}
        input_check(kwargs, 'sigma_ratio', 1.6, self.params, exception=False)
        input_check(kwargs, 'overlap', 0.5, self.params, exception=False)
        input_check(kwargs, 'threshold_rel', None, self.params, exception=False)
        input_check(kwargs, 'exclude_border', False, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):
        """
        Applies a canny filter to the input images

        :param in_imw: Images of the shape (N, W, H)
        :return:
        """
        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        out_img = self.apply(in_imw.images)

        # If it is the output of a different function then take the last value in the tuple
        category = f'\n Difference of Gaussian'
        if self.params:
            category += f' and params: {self.params}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs):
        """
        Applies a canny filter, essentially a wrapper for the scikit-image.feature.canny() method.

        To do: not vectorised. Perhaps can be implemented with joblib?
        https://scikit-image.org/docs/stable/user_guide/tutorial_parallelization.html

        """
        # For loop to apply the canny function to each img in the image array
        out_list = [sk_blob_dog(x, **self.params) for x in in_imgs]
        out_imgs = np.stack(out_list, axis=0)

        return out_imgs

if __name__ == '__main__':

    do_kernel = True
    if do_kernel:
        n_samples = 1001
        images = DefectViewer() << (ImageLoader(defect_class='FrontGridInterruption') << n_samples)

        # ck = CreateKernel(bim=2, kernel='gaussian', size=3, std=8)
        start = time.perf_counter()
        c_imgs = HOG(pixels_per_cell=(3, 3), num_jobs=20) << images

        print(time.perf_counter() - start)

        _ = Show('hog', num_images=10) << c_imgs
