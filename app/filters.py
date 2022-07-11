import numpy as np
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve, convolve1d
from skimage.feature import hog as sk_hog
from skimage.feature import canny as sk_canny
from app.imager import ImageLoader, DefectViewer, Show


def input_check(indict, key, default, out_dict, exception=False):
    try:
        out_dict[key] = indict[key]
        del indict[key]
    except KeyError:
        if exception:
            raise KeyError(f'{key} is a required input and was not provided')
        else:
            if default is not None:
                out_dict[key] = default


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
        self.kernel_params = kwargs

        # Create a gaussian kernel
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
            self.kernel_params = kwargs
            self.kernel_val = self.kernel_params['custom_kernel']

        else:
            raise KeyError('Kernel type not recognised. Allowable values: gaussian, custom, prewitt, sobel')

    def get(self):

        return self.kernel_val

    def __lshift__(self, in_imgs):
        """
        Applies a kernel to images and returns that image with the kernel applied.
        :param in_imgs:
        :return:
        """

        # If it is the output of a different function then take the last value in the tuple
        if isinstance(in_imgs, tuple):
            in_imgs = in_imgs[-1]

        return in_imgs, self.kernel_val

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

    def __init__(self, **kwargs):
        """

        :param mode: Convolution model.
                    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html
        :param cval: If mode is constant then this value will be used.
        :param axis or axes: The axis of input along which to convolve.
                    Default is -1 for one dimensional and (-2, -1) for 2D
        """

        try:
            self.axis = kwargs['axis']
        except KeyError:
            try:
                self.axis = kwargs['axes']
            except KeyError:
                self.axis = None

        # Mode for treating the edges
        try:
            self.mode = kwargs['mode']
        except KeyError:
            self.mode = 'reflect'

        self.cval = 0.0
        if self.mode == 'constant':
            try:
                self.cval = kwargs['cval']
            except KeyError:
                pass

    def __lshift__(self, kern_out):
        """
        Applies a kernel to images and returns that image with the kernel applied.
        :param kern_out: Output of the Kernel class (images, kernel)
        :return:
        """

        in_imgs, kernel = kern_out

        # If it is the output of a different function then take the last value in the tuple
        if isinstance(in_imgs, tuple):
            in_imgs = in_imgs[-1]

        return in_imgs, self.apply_filter(in_imgs, kernel)

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

    def __init__(self, sigma, **kwargs):
        """

        :param sigma:
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

        if kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imgs):
        """
        Applies a canny filter, essentially a wrapper for the scikit-image.feature.canny() method.

        :param in_imgs: Images of the shape (N, W, H)
        :return:
        """

        # If it is the output of a different function then take the last value in the tuple
        if isinstance(in_imgs, tuple):
            in_imgs = in_imgs[-1]

        return in_imgs, self.apply_filter(in_imgs)

    def apply_filter(self, in_imgs):
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

    def __init__(self, **kwargs):
        """
        Note the following parameters are not configurable given our datapoints:
        - visualize = True: returns the image of the HOG
        - multichannel = False: does not allow multichannel images.
        - channel_axis = None: does not allow specification of the channel axis.

        :param orientation: number of orientation bins.
        :param pixels_per_cell: tupel of the number of cells per block.

        :return:
        """
        # orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (3, 3),
        # block_norm = 'L2-Hys', transform_sqrt = False, feature_vector = True

        self.params = {'visualize': True}
        input_check(kwargs, 'orientations', 9, self.params, exception=False)
        input_check(kwargs, 'pixels_per_cell', (8, 8), self.params, exception=False)
        input_check(kwargs, 'cells_per_block', (3, 3), self.params, exception=False)
        input_check(kwargs, 'block_norm', 'L2-Hys', self.params, exception=False)
        input_check(kwargs, 'transform_sqrt', False, self.params, exception=False)
        input_check(kwargs, 'feature_vector', True, self.params, exception=False)

        if kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imgs):
        """
        Applies a filter to images and returns that image with the filter applied.

        :param in_imgs: Output of the Kernel class (images, kernel)
        :return: original images, the filtered image.
        """

        # If it is the output of a different function then take the last value in the tuple
        if isinstance(in_imgs, tuple):
            in_imgs = in_imgs[-1]

        return in_imgs, self.apply_filter(in_imgs)

    def apply_filter(self, in_imgs):
        """
        Applies a canny filter, essentially a wrapper for the scikit-image.feature.hog() method.

        To do: not vectorised. Perhaps can be implemented with joblib?
        https://scikit-image.org/docs/stable/user_guide/tutorial_parallelization.html

        """
        # For loop to apply the HOG filter to each img in the image array
        # the sk_hog method returns a tuple, and we are only interested in the image,
        # which is the second element.
        # For loop to apply the canny function to each img in the image array
        out_list = [sk_hog(x, **self.params)[1] for x in in_imgs]
        out_imgs = np.stack(out_list, axis=0)

        return out_imgs


if __name__ == '__main__':

    do_kernel = True
    if do_kernel:
        n_samples = 10
        images = DefectViewer() << (ImageLoader(defect_class='FrontGridInterruption') << n_samples)

        # ck = CreateKernel(bim=2, kernel='gaussian', size=3, std=8)
        c_imgs = HOG() << images

        _ = Show('hog') << c_imgs
