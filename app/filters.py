import copy
import time
import math
import numpy as np
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve, convolve1d
from skimage.feature import hog as sk_hog
from skimage.feature import canny as sk_canny
from skimage.filters import meijering as sk_meijering
from skimage.filters import frangi as sk_frangi
from skimage.filters import hessian as sk_hessian
from skimage.filters import sato as sk_sato
from skimage.filters import threshold_multiotsu as sk_threshold_multiotsu
from skimage.filters import farid as sk_farid
from skimage.filters import farid_v as sk_farid_v
from skimage.filters import farid_h as sk_farid_h
from app.imager import ImageLoader, DefectViewer, Show
from app.utils import input_check, ImageWrapper, line_split_string, parallelize


class CreateKernel:
    """
    Creates and applies a filter to an input image.
    """

    def __init__(self, dim=2, kernel='gaussian', **kwargs):
        """

        :param dim:
        :param kernel: Type of kernel to create options include:
            'gaussian': gaussian kernel
            'sobel': sobel filter
            'prewitt filter':
            'custom': custom specified kernel
        :return:
        """

        self.dim = dim
        self.missing_list = []

        # Create a gaussian kernel
        self.kernel_params = kwargs
        if kernel == 'gaussian':
            # Check all parameters are passed
            self.required_params = ['size', 'std']
            self._check_required_params()

            self.kernel_type = kernel
            self.kernel_val = self.generate_gaussian_kernel()

        # Create a prewitt filter
        elif kernel == 'prewitt':
            # Check all parameters are passed
            self.required_params = ['axis']
            self._check_required_params()

            self.kernel_type = kernel
            self.kernel_val = self.generate_prewitt_kernel()

        # Create a sobel filter
        elif kernel == 'sobel':
            # Check all parameters are passed
            self.required_params = ['axis']
            self._check_required_params()

            self.kernel_type = kernel
            self.kernel_val = self.generate_sobel_kernel()

        # pass a custom kernel
        elif kernel == 'custom':
            # Check all parameters are passed
            self.required_params = ['custom_kernel']
            self._check_required_params()

            self.kernel_type = kernel
            self.kernel_val = self.kernel_params['custom_kernel']
            del self.kernel_params['custom_kernel']

        else:
            raise KeyError('Kernel type not recognised. Allowable values: gaussian, custom, prewitt, sobel')

    def apply(self):

        return self.kernel_val

    def __lshift__(self, in_imw):
        """
        Applies a kernel to images and returns that image with the kernel applied.
        :param in_imw:
        :return:
        """

        # If it is the output of a different function then take the last value in the tuple
        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        if self.kernel_params:
            category = f'\n kernel type {self.kernel_type} with params {self.kernel_params}'
        else:
            category = f'\n kernel type {self.kernel_type}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(in_imw.images, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return out_imw, self.kernel_val

    def _check_required_params(self):
        """
        Checks that all the required kwargs have been passed
        :param :
        :return:
        """

        for param in self.required_params:
            if param not in self.kernel_params.keys():
                self.missing_list.append(param)
            else:
                pass

        # Check all required parameters are there:
        if len(self.missing_list) > 0:
            raise KeyError(f'Missing required parameters: {self.missing_list}')

    def generate_gaussian_kernel(self):
        """
        Generates a gaussian kernel. Requires the size and std to be parsed during
        instantiation as part of key-value kwargs.

        :param :
        :return: numpy array with the filter
        """
        # Extract size and sigma
        size = self.kernel_params['size']
        std = self.kernel_params['std']

        # Create 1d kernel
        gaussian_1d = gaussian(size, std, sym=True)

        if self.dim == 1:
            return gaussian_1d / gaussian_1d.sum()

        elif self.dim == 2:
            # Create 2d filter and normalise
            gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
            return gaussian_2d / gaussian_2d.sum()

    def generate_sobel_kernel(self):
        """
        Generates a sobel kernel, requires the 'axis' argument to be parsed during
        instantiation.
        ** Note by definition, this is a 2D kernel, thus if the dimension argument is
        1 this will be overridden to a 2D kernel. **

        :param :
        :return: numpy array with the filter
        """
        # Extract axis
        axis = self.kernel_params['axis']

        if self.dim == 1:
            print('Warning, forcing 2D dimensionality, despite 1D being specified')
            self.dim = 2

        # Check axis and create filter
        if int(axis) not in (0, 1):
            raise ValueError(f'for a 2D sobel filter, axis must be equal to 0 or 1 but \'{axis}\' was provided.')
        elif int(axis) == 0:
            return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        elif int(axis) == 1:
            return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).T

    def generate_prewitt_kernel(self):
        """
        Generates a prewitt kernel, requires the 'axis' argument to be parsed during
        instantiation.
        ** Note by definition, this is a 2D kernel, thus if the dimension argument is
        1 this will be overridden to a 2D kernel. **

        :param :
        :return: numpy array with the filter
        """
        # Extract axis
        axis = self.kernel_params['axis']

        if self.dim == 1:
            print('Warning, forcing 2D dimensionality, despite 1D being specified')
            self.dim = 2

        # Check axis and create filter
        if int(axis) not in (0, 1):
            raise ValueError(f'for a 2D prewitt filter, axis must be equal to 0 or 1 but \'{axis}\' was provided.')
        elif int(axis) == 0:
            return np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        elif int(axis) == 1:
            return np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]).T


class Convolve:
    """
    Convolves a kernel with the array
    """

    def __init__(self, consume_kwargs=True, **kwargs):
        """

        :param consume_kwargs: If True check for unused Kwargs
        :param mode: Convolution model.
                    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html
        :param cval: If mode is constant then this value will be used.
        :param axis or axes: The axis of input along which to convolve.
                    Default is -1 for one dimensional and (-2, -1) for 2D
        """

        try:
            self.axis = kwargs['axis']
            del kwargs['axis']
        except KeyError:
            try:
                self.axis = kwargs['axes']
                del kwargs['axes']
            except KeyError:
                self.axis = None

        try:
            self.num_jobs = kwargs['num_jobs']
            del kwargs['num_jobs']
        except KeyError:
            self.num_jobs = 1

        # Mode for treating the edges
        try:
            self.mode = kwargs['mode']
            del kwargs['mode']
        except KeyError:
            self.mode = 'reflect'

        self.cval = 0.0
        if self.mode == 'constant':
            try:
                self.cval = kwargs['cval']
                del kwargs['cval']
            except KeyError:
                pass

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, kern_out):
        """
        Applies a kernel to images and returns that image with the kernel applied.
        :param kern_out: Output of the Kernel class (images, kernel)
        :return:
        """

        in_imw, kernel = kern_out
        out_img = self.apply(in_imw.images, kernel)

        category = f' convolved along axis {self.axis}'
        category = in_imw.category + line_split_string(category)
        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs, kernel):
        """

        :return:
        """
        if self.num_jobs == 1:
            return self.apply_filter(in_imgs, kernel)
        else:
            # Divide in_imgs into chunks
            # At least 2 images per job
            num_jobs = self.num_jobs
            chunk_size = math.ceil(in_imgs.shape[0] / num_jobs)
            chunk_size = 1 if chunk_size < 1 else chunk_size

            # Split the image into so many chunks
            args = [(in_imgs[i:i + chunk_size, :], kernel) for i in range(0, in_imgs.shape[0], chunk_size)]
            funcs = [self.apply_filter for _ in range(len(args))]

            # Collect the results from parallelize
            results = parallelize(funcs, args)

            # Concatenate and return results
            return np.concatenate(results, axis=0)

    def apply_filter(self, in_imgs, kernel):
        """
        Wrapper for the scipy.signal.convolve2d method:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html#scipy.signal.convolve2d
        Wrapper for the scipy.signal.convolve1d method:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html

        :param in_imgs: The array to apply the filter to
        :param kernel: Kernel weights
        :return: array with the filter applied
        """

        # Number of dimensions in convolution
        dim = np.sum([x > 1 for x in kernel.shape])
        if dim <= 1:
            dim = 1
            kernel = kernel.flatten()

            axis = -1 if self.axis is None else self.axis
        else:
            axis = (-2, -1) if self.axis is None else self.axis

            # Expand the kernel to the dimensions of interest
            for _ in range(len(in_imgs.shape) - 2):
                kernel = kernel[np.newaxis, :]

        if dim == 1:
            return convolve1d(in_imgs, kernel, mode=self.mode, axis=axis)
        elif dim == 2:
            # Create 2d filter and normalise
            return convolve(in_imgs, kernel, mode=self.mode, cval=self.cval)


class Canny:
    """
    Applies a canny filter to the image
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html
    """

    def __init__(self, sigma, consume_kwargs=True, **kwargs):
        """

        :param sigma:
        :param consume_kwargs: If True check for empty kwargs
        :param **kwargs: see below

        :keyword Arguments:
            :low_threshold:
            :high_threshold:
            :mask:
            :use_quantiles:
            :mode:
            :cval:
        """

        self.params = {'sigma': sigma}
        input_check(kwargs, 'low_threshold', None, self.params, exception=False)
        input_check(kwargs, 'high_threshold', None, self.params, exception=False)
        input_check(kwargs, 'mask', None, self.params, exception=False)
        input_check(kwargs, 'use_quantiles', False, self.params, exception=False)
        input_check(kwargs, 'mode', 'constant', self.params, exception=False)
        input_check(kwargs, 'cval', 0.0, self.params, exception=False)

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
        category = f'\n Canny with sigma'
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
        out_list = [sk_canny(x, **self.params) for x in in_imgs]
        out_imgs = np.stack(out_list, axis=0)

        return out_imgs


class HOG:
    """
    Extracts a histogram of orineted gradients for an input image.
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog

    """

    def __init__(self, consume_kwargs=True, **kwargs):
        """
        Note the following parameters are not configurable given our datapoints:
        - visualize = True: returns the image of the HOG
        - multichannel = False: does not allow multichannel images.
        - channel_axis = None: does not allow specification of the channel axis.

        :param consume_kwargs: If True then check for empty kwargs
        :param pixels_per_cell: tuple of the number of cells per block.

        :return:
        """
        # orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (3, 3),
        # block_norm = 'L2-Hys', transform_sqrt = False, feature_vector = True

        self.params = {'visualize': True}
        self.num_jobs = {}
        input_check(kwargs, 'orientations', 9, self.params, exception=False)
        input_check(kwargs, 'pixels_per_cell', (8, 8), self.params, exception=False)
        input_check(kwargs, 'cells_per_block', (3, 3), self.params, exception=False)
        input_check(kwargs, 'block_norm', 'L2-Hys', self.params, exception=False)
        input_check(kwargs, 'transform_sqrt', False, self.params, exception=False)
        input_check(kwargs, 'feature_vector', True, self.params, exception=False)
        input_check(kwargs, 'num_jobs', 1, self.num_jobs, exception=False)
        self.num_jobs = self.num_jobs['num_jobs']

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):
        """
        Applies a HOG filter to the input images.

        :param in_imw: Images of the shape (N, W, H)
        :return:
        """

        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        out_img = self.apply(in_imw.images)

        # If it is the output of a different function then take the last value in the tuple
        category = f'\n HOG filter '
        if self.params:
            category += f' with params: {self.params}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs, get_images=True):
        """

        :return:
        """
        if self.num_jobs == 1:
            return self.apply_filter(in_imgs)
        else:
            # Divide in_imgs into chunks
            # At least 2 images per job
            num_jobs = self.num_jobs
            chunk_size = math.ceil(in_imgs.shape[0] / num_jobs)
            chunk_size = 1 if chunk_size < 1 else chunk_size

            # Split the image into so many chunks
            args = [(in_imgs[i:i + chunk_size, :], get_images) for i in range(0, in_imgs.shape[0], chunk_size)]
            funcs = [self.apply_filter for _ in range(len(args))]

            # Collect the results from parallelize
            results = parallelize(funcs, args)

            if get_images:
                # Concatenate and return results
                return np.concatenate(results, axis=0)
            else:
                return results

    def apply_filter(self, in_imgs, get_images=True):
        """
        Applies a HOG filter, essentially a wrapper for the scikit-image.feature.hog() method.

        """
        # For loop to apply the HOG filter to each img in the image array
        # the sk_hog method returns a tuple, and we are only interested in the image,
        # which is the second element.
        # For loop to apply the function to each img in the image array
        if get_images:
            out_list = [sk_hog(x, **self.params)[1] for x in in_imgs]
            out_imgs = np.stack(out_list, axis=0)

            return out_imgs
        else:
            return [sk_hog(x, **self.params)[0] for x in in_imgs]


class Meijering:
    """
    Applies a meijering filter to the image
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.meijering
    """

    def __init__(self, sigmas, consume_kwargs=True, **kwargs):
        """
        default sigmas=range(1,10,2)
        :param sigmas:
        :param consume_kwargs: If True check for empty kwargs
        :param **kwargs: see below

        :keyword Arguments:
            :mode:
            :alpha:
            :black_ridges:
            :cval:
        """

        self.params = {'sigmas': sigmas}
        input_check(kwargs, 'mode', 'wrap', self.params, exception=False)
        input_check(kwargs, 'alpha', None, self.params, exception=False)
        input_check(kwargs, 'black_ridges', True, self.params, exception=False)
        input_check(kwargs, 'cval', 0.0, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):
        """
        Applies a meijering neuriteness filter to the input images

        :param in_imw: Images of the shape (N, W, H)
        :return:
        """
        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        out_img = self.apply(in_imw.images)

        # If it is the output of a different function then take the last value in the tuple
        category = f'\n Meijering filter'
        if self.params:
            category += f' and params: {self.params}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs):
        """
        Applies a meijering filter
        """
        # For loop to apply the function to each img in the image array
        out_list = [sk_meijering(x, **self.params) for x in in_imgs]
        out_imgs = np.stack(out_list, axis=0)

        return out_imgs


class Frangi:
    """
    Applies a Frangi filter to the image
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.frangi
    """

    def __init__(self, sigmas, consume_kwargs=True, **kwargs):
        """
        default: sigmas=range(1,10,2)
        :param sigmas:
        :param consume_kwargs: If True check for empty kwargs
        :param **kwargs: see below

        :keyword Arguments:
            :scale_range:
            :scale_step:
            :alpha:
            :beta:
            :gamma:
            :mode:
            :alpha:
            :black_ridges:
            :cval:
        """

        self.params = {'sigmas': sigmas}
        input_check(kwargs, 'scale_range', None, self.params, exception=False)
        input_check(kwargs, 'scale_step', None, self.params, exception=False)
        input_check(kwargs, 'alpha', 0.5, self.params, exception=False)
        input_check(kwargs, 'beta', 0.5, self.params, exception=False)
        input_check(kwargs, 'gamma', 15, self.params, exception=False)
        input_check(kwargs, 'black_ridges', True, self.params, exception=False)
        input_check(kwargs, 'mode', 'wrap', self.params, exception=False)
        input_check(kwargs, 'cval', 0.0, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):
        """
        Applies a frangi vesselness filter to the input images

        :param in_imw: Images of the shape (N, W, H)
        :return:
        """
        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        out_img = self.apply(in_imw.images)

        # If it is the output of a different function then take the last value in the tuple
        category = f'\n Frangi filter'
        if self.params:
            category += f' and params: {self.params}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs):
        """
        Applies a Frangi filter
        """
        # For loop to apply the function to each img in the image array
        out_list = [sk_frangi(x, **self.params) for x in in_imgs]
        out_imgs = np.stack(out_list, axis=0)

        return out_imgs


class Hessian:
    """
    Applies a Hessian filter to the image
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.hessian
    """

    def __init__(self, sigmas, consume_kwargs=True, **kwargs):
        """
        sigmas=range(1,10,2)
        :param sigmas:
        :param consume_kwargs: If True check for empty kwargs
        :param **kwargs: see below

        :keyword Arguments:
            :scale_range:
            :scale_step:
            :alpha:
            :beta:
            :gamma:
            :mode:
            :alpha:
            :black_ridges:
            :cval:
        """

        self.params = {'sigmas': sigmas}
        input_check(kwargs, 'scale_range', None, self.params, exception=False)
        input_check(kwargs, 'scale_range', None, self.params, exception=False)
        input_check(kwargs, 'scale_step', None, self.params, exception=False)
        input_check(kwargs, 'beta', 0.5, self.params, exception=False)
        input_check(kwargs, 'gamma', 15, self.params, exception=False)
        input_check(kwargs, 'black_ridges', True, self.params, exception=False)
        input_check(kwargs, 'mode', 'wrap', self.params, exception=False)
        input_check(kwargs, 'cval', 0.0, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):
        """
        Applies a Hessian vesselness filter to the input images

        :param in_imw: Images of the shape (N, W, H)
        :return:
        """
        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        out_img = self.apply(in_imw.images)

        # If it is the output of a different function then take the last value in the tuple
        category = f'\n Hessian filter'
        if self.params:
            category += f' and params: {self.params}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs):
        """
        Applies a Hessian filter
        """
        # For loop to apply the function to each img in the image array
        out_list = [sk_hessian(x, **self.params) for x in in_imgs]
        out_imgs = np.stack(out_list, axis=0)

        return out_imgs


class Sato:
    """
    Applies a Sato filter to the image
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sato
    """

    def __init__(self, sigmas, consume_kwargs=True, **kwargs):
        """
        sigmas=range(1,10,2)
        :param sigmas:
        :param consume_kwargs: If True check for empty kwargs
        :param **kwargs: see below

        :keyword Arguments:
            :sigmas:
            :mode:
            :black_ridges:
            :cval:
        """

        self.params = {'sigmas': sigmas}
        input_check(kwargs, 'black_ridges', True, self.params, exception=False)
        input_check(kwargs, 'mode', 'wrap', self.params, exception=False)
        input_check(kwargs, 'cval', 0.0, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):
        """
        Applies a Sato tubeness filter to the input images

        :param in_imw: Images of the shape (N, W, H)
        :return:
        """
        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        out_img = self.apply(in_imw.images)

        # If it is the output of a different function then take the last value in the tuple
        category = f'\n Sato filter'
        if self.params:
            category += f' and params: {self.params}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs):
        """
        Applies a Sato filter
        """
        # For loop to apply the function to each img in the image array
        out_list = [sk_sato(x, **self.params) for x in in_imgs]
        out_imgs = np.stack(out_list, axis=0)

        return out_imgs


class Farid:
    """
    Applies a Farid filter to the image
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid
    """

    def __init__(self, how='default', consume_kwargs=True, **kwargs):
        """
        sigmas=range(1,10,2)
        :param how: orientation insensitive, horizontal or vertical
        :param consume_kwargs: If True check for empty kwargs
        :param **kwargs: see below

        :keyword Arguments:
            :mask: Optional mask to limit the application area
        """

        self.params = {'how': how}
        input_check(kwargs, 'mask', None, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):
        """
        Applies a farid filter to the images

        :param in_imw: Images of the shape (N, W, H)
        :return:
        """
        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        out_img = self.apply(in_imw.images)

        # If it is the output of a different function then take the last value in the tuple
        category = f'\n Farid filter'
        if self.params:
            category += f' and params: {self.params}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs):
        """
        Applies a Farid filter
        """
        # For loop to apply the function to each img in the image array
        if self.params['how'] == 'default':
            out_list = [sk_farid(x) for x in in_imgs]
        elif self.params['how'] == 'horizontal':
            out_list = [sk_farid_h(x) for x in in_imgs]
        elif self.params['how'] == 'vertical':
            out_list = [sk_farid_v(x) for x in in_imgs]
        else:
            raise KeyError(f"Unsupported fileter type {self.params['how']}")

        out_imgs = np.stack(out_list, axis=0)

        return out_imgs


class ThresholdMultiotsu:
    """
    Generates n_classes-1 threshold values to divide the image and applies it. 
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_multiotsu
    follows implementation guide in
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html#sphx-glr-auto-examples-segmentation-plot-multiotsu-py
    """

    def __init__(self, consume_kwargs=True, **kwargs):
        """
        :param classes: Number of classes to divide levels 
        :param consume_kwargs: If True check for empty kwargs
        :param **kwargs: see below

        :keyword Arguments:
            :nbins:
            :hist:
        """
        self.levels = []

        if 'classes' in kwargs:
            self.classes = kwargs['classes']
            del kwargs['classes']
        else:
            self.classes = 3

        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
            del kwargs['threshold']
        else:
            self.threshold = 1

        if 'digitize' in kwargs:
            self.digitize = kwargs['digitize']
            del kwargs['digitize']
        else:
            self.digitize = True

        self.params = {}

        input_check(kwargs, 'nbins', 256, self.params, exception=False)
        input_check(kwargs, 'hist', None, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):
        """
        Calculates the Multi-Otsu thresholds and applies it to the image.

        :param in_imw: Images of the shape (N, W, H)
        :return:
        """
        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        out_img = self.apply(in_imw.images)

        # If it is the output of a different function then take the last value in the tuple
        category = f'\n Multi-Otsu threshold filter into {self.classes} classes'
        if self.params:
            category += f' and params: {self.params}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs, return_rejects=False):
        """
        Applies a multiotsu threshold filter to find the thresholds, then applies
        the thresholding to the image via np.digitize. 
        """
        # For loop to apply the function to each img in the image array
        out_list = []

        self.check_thresholds()

        keep = []
        for cnt, img in enumerate(in_imgs):

            # Get thresholds and save
            try:
                levels = sk_threshold_multiotsu(img, classes=self.classes, **self.params)
            except ValueError:
                print(f'Failed on count {cnt}')
                continue

            # These are the indices to keep
            keep.append(cnt)

            self.levels.append(levels)

            # apply thresholding to image
            if self.digitize:
                out_list.append(np.digitize(img, bins=levels))
            else:
                mask = (img > levels[self.threshold - 1])
                out_list.append(img * mask)

        out_imgs = np.stack(out_list, axis=0)

        if not return_rejects:
            return out_imgs
        else:
            print(f'{in_imgs.shape[0] - len(keep)} images were rejected')
            return out_imgs, keep

    def check_thresholds(self):

        if self.threshold - 1 < 0:
            raise Exception('The selected threshold must be greater than 0.')
        elif self.threshold >= self.classes:
            raise Exception('The selected threshold must be lower than the number of classes.')


if __name__ == '__main__':

    do_kernel = True
    if do_kernel:
        n_samples = 1001
        images = DefectViewer() << (ImageLoader(defect_class='FrontGridInterruption') << n_samples)

        # ck = CreateKernel(dim=2, kernel='gaussian', size=3, std=8)
        start = time.perf_counter()
        c_imgs = HOG(pixels_per_cell=(3, 3), num_jobs=20) << images

        print(time.perf_counter() - start)

        _ = Show('hog', num_images=10) << c_imgs
