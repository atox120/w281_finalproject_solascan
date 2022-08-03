import copy
import numpy as np
from app.utils import ImageWrapper
from app.custom import RemoveBusBars, Orient
from app.imager import Exposure, ImageLoader, DefectViewer
from app.filters import CreateKernel, Convolve, ThresholdMultiotsu, Sato
from app.transforms import FFT, IFFT, Butterworth


def get_samples(defect_classes, num_samples, complimentary=True):
    """

    :param defect_classes: These are the classes for this model to use
    :param num_samples: Number of samples of the classes
    :param complimentary:
    :return:
    """
    # Load 10 examples and name the category for it. Category is like a title for images
    defect_class = defect_classes
    defect = (DefectViewer(row_chop=15, col_chop=15) <<
              (ImageLoader(defect_class=defect_class, seed=20) << num_samples))
    defect.category = str(defect_classes)

    # Make the other teh same length as the defect
    num_samples = len(defect)

    # Get the not this defect
    if complimentary:
        not_defect = (DefectViewer(row_chop=15, col_chop=15) << (
                ImageLoader(defect_class=defect_class, is_not=True) << num_samples * 2))
        not_defect.category = 'Others'
    else:
        not_defect = (DefectViewer(row_chop=15, col_chop=15) << (ImageLoader(defect_class='None') << num_samples * 2))
        not_defect.category = 'None'

    # Create a copy of the defect
    defect_ = defect.copy()

    # Eliminate any not defect images that are in defect
    defect = defect - not_defect

    # ELiminate any defect images that are in not defect
    not_defect = not_defect - defect_
    return defect, not_defect


def get_data_handler(defect_classes):
    if 'FrontGridInterruption' in defect_classes:
        print('model_features.grid_interruption')
        data_handler = grid_interruption
    elif 'Closed' in defect_classes:
        print('model_features.closed')
        data_handler = closed
    elif 'Isolated' in defect_classes:
        print('model_features.isolated')
        data_handler = isolated
    elif 'BrightSpot' in defect_classes or 'Corrosion' in defect_classes:
        print('model_features.generic_return')
        data_handler = generic_return
    else:
        raise KeyError('Unsupported model type')

    return data_handler


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


def brightspots(in_imw, num_jobs=20):
    """
    """

    if isinstance(in_imw, ImageWrapper):
        images = ~in_imw
    else:
        images = in_imw

    # Re orient the images that are off by 90
    oriented_images = Orient(num_jobs=num_jobs, do_debug=False, do_eliminate=False).apply(images)[0]

    # Create a Gaussian blur kernel and Convolve 
    gaussian_kernel = CreateKernel(kernel='gaussian', size=5, std=8).apply()
    gaussian_images = Convolve().apply(oriented_images, gaussian_kernel)
    
    return_images = np.concatenate((images, gaussian_images), axis=-1)

    #fourier_defect = (FFT(dim=2) << defect_blur) 
    fft_images = FFT(dim=2).apply(gaussian_images)
    #return_images = IFFT(mask=bandpass).apply(fft_images)
    return_images = np.concatenate((images, fft_images[-2]), axis=-1)

    if isinstance(in_imw, ImageWrapper):
        return ImageWrapper(return_images, category=in_imw.category + '\n Brightspots - Gaussian Blur - Fourier Transform',
                            image_labels=copy.deepcopy(in_imw.image_labels))

    # The and operator con
    return return_images




def closed(in_imw, num_jobs=None):
    """

    :return:
    """
    if num_jobs:
        pass

    if isinstance(in_imw, ImageWrapper):
        images = ~in_imw
    else:
        images = in_imw

    # Pipeline for Closed Cracks
    sato_filtered = Sato(sigmas=[1, 2]).apply(images)
    stretched_images = Exposure('stretch').apply(sato_filtered)
    return_images = ThresholdMultiotsu(n_classes=4, threshold=1, digitize=False).apply(stretched_images)

    if isinstance(in_imw, ImageWrapper):
        return ImageWrapper(return_images, category=in_imw.category + '\n Closed - Preprocessed',
                            image_labels=copy.deepcopy(in_imw.image_labels))

    return return_images


def isolated(in_imw, num_jobs=None):
    """

    :return:
    """
    if num_jobs:
        pass

    if isinstance(in_imw, ImageWrapper):
        images = ~in_imw
    else:
        images = in_imw

    # Apply Isolated Pipeline
    inverted_images = Exposure('invert').apply(images)
    return_images = ThresholdMultiotsu(n_classes=2).apply(inverted_images)

    if isinstance(in_imw, ImageWrapper):
        return ImageWrapper(return_images, category=in_imw.category + '\n Closed - Preprocessed',
                            image_labels=copy.deepcopy(in_imw.image_labels))

    return return_images


def generic_return(in_imw, num_jobs=None):
    """

    :return:
    """
    if num_jobs:
        pass

    if isinstance(in_imw, ImageWrapper):
        return_images = ~in_imw
    else:
        return_images = in_imw

    if isinstance(in_imw, ImageWrapper):
        return ImageWrapper(return_images, category=in_imw.category + f'\n processed',
                            image_labels=copy.deepcopy(in_imw.image_labels))
    return return_images


def resistive_crack(in_imw, num_jobs=20):
    """

    """

    # Grab the images, if image wrapper unwrap
    if isinstance(in_imw, ImageWrapper):
        images = ~in_imw
    else:
        images = in_imw

    # Reorient the image
    oriented_images = Orient(num_jobs=num_jobs).apply(images)[0]

    # Remove Busbars
    nobus_images = RemoveBusBars(num_jobs=num_jobs).apply(in_imgs=oriented_images, in_hogs=None, do_debug=False)

    # Creat Butterworth Bandpass
    bandpass = Butterworth(nobus_images[-1]).bandpass(3, 15, 1, 1)

    # Apply fft and ifft
    fft_tuple = FFT(dim=2, axis=(-2, -1)).apply(nobus_images)
    return_images = IFFT(mask=bandpass).apply(fft_tuple)

    # Rewrap immages (if applicable)
    if isinstance(in_imw, ImageWrapper):
        return ImageWrapper(return_images, category=in_imw.category + '\n ResistiveCrack - Preprocessed',
                            image_labels=copy.deepcopy(in_imw.image_labels))

    return return_images
