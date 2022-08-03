import copy
import numpy as np
from app.custom import Orient
from app.utils import ImageWrapper
from app.filters import CreateKernel, Convolve


def grid_interruption(in_imw, num_jobs=20):
    """
    """

    if isinstance(in_imw, ImageWrapper):
        images = ~in_imw
    else:
        images = in_imw

    # Re orient the images that are off by 90
    oriented_images = Orient(num_jobs=num_jobs, do_debug=False, do_eliminate=False).apply(images)[0]

    # Create a sobel kernel and Convolve
    sobel_kernel = CreateKernel(kernel='sobel', axis=0).apply()
    sobel_images = Convolve().apply(oriented_images, sobel_kernel)

    return_images = np.concatenate((images, sobel_images), axis=-1)

    if isinstance(in_imw, ImageWrapper):
        return ImageWrapper(return_images, category=in_imw.category + '\n GridInterruption - Preprocessed',
                            image_labels=copy.deepcopy(in_imw.image_labels))

    # The and operator con
    return return_images
