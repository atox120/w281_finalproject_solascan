import cv2
import numpy as np

from collections.abc import Iterable
from scipy.ndimage import convolve, convolve1d
from scipy.signal.windows import gaussian
from skimage.feature import canny as sk_canny

from app.imager import ImageLoader, DefectViewer


class FFT:

    def __init__(self, dim=2, axis=(-2, -1), return_which='both'):
        """

         :param dim:
         :param axis: Which dimension(s) to perform FFT on.
                When dim is 2 then axis should be a tuple. Default is the last two dimensions.
                When dim is 1 then axis should be an integer dimension.
         :param return_which:
                If 'both' (default) then returns (orig_img, magnitude, phase).
                If 'magnitude' then return only (orig_img, , magnitude).
                If 'phase' then return only (orig_img, , phase)
        """

        self.dim = dim
        self.axis = axis
        self.return_which = return_which

        if self.dim == 1:
            if isinstance(axis, Iterable):
                raise TypeError('A single dimension FFT can only be done on one axis')

    def __lshift__(self, in_img):
        """
        Applies an FFT transform to an image and returns an image of the same size

        :param in_img:
        :return:
        """
        # in_img = 1 - (in_img/255.0)

        if len(in_img.shape) == 2:
            in_img = in_img[np.newaxis, :, :]

        # Create a window function
        win = self.create_window(in_img)

        if self.dim == 2:
            return self.fft2(in_img, win)
        else:
            return self.fft(in_img, win)

    @staticmethod
    def create_window(in_img):
        # Create a window function
        win = np.outer(np.hanning(in_img.shape[-2]), np.hanning(in_img.shape[-1]))
        win = win / np.mean(win)

        # Make the dimensions such that it works for the element wise multiplication
        for _ in range(len(in_img.shape)-2):
            win = win[np.newaxis, :]

        return win

    # Display the fft and the image
    def fft2(self, in_img, win):
        """

        :param in_img: Grey scale images to be transformed of the shape (N, height, width)
        :param win: Windowing function array
        :return:
        """

        # 2D fourier transform
        transformed = np.fft.fftshift(np.fft.fft2(in_img * win, axes=self.axis), axes=self.axis)

        if self.return_which == 'both':
            magnitude = np.log10(np.abs(transformed))
            phase = np.angle(transformed)
            return in_img, magnitude, phase
        elif self.return_which == 'magnitude':
            magnitude = np.log10(np.abs(transformed))
            return in_img, magnitude
        elif self.return_which == 'phase':
            phase = np.angle(transformed)
            return in_img, phase
        else:
            raise TypeError('return_which must be one of magnitude, phase or both')

    # Display the fft and the image
    def fft(self, in_img, win):
        """

        :param in_img: Grey scale images to be transformed of the shape (N, height, width)
        :param win: Windowing function array
        :return:
        """

        # 2D fourier transform
        transformed = np.fft.fftshift(np.fft.fft(in_img * win, axis=self.axis), axes=self.axis)
        if self.return_which == 'both':
            magnitude = np.log10(np.abs(transformed))
            phase = np.angle(transformed)
            return in_img, magnitude, phase
        elif self.return_which == 'magnitude':
            magnitude = np.log10(np.abs(transformed))
            return in_img, magnitude
        elif self.return_which == 'phase':
            phase = np.angle(transformed)
            return in_img, phase
        else:
            raise TypeError('return_which must be one of magnitude, phase or both')


class IFFT:

    def __init__(self, mask=None, axis=(-2, -1)):
        """
        :param mask: A mask for masking out the
        """

        if mask is not None and len(mask.shape) == 2:
            mask = mask[np.newaxis, :]

        self.mask = mask
        self.axis = axis

    def __lshift__(self, fft_out):
        """
        Applies an FFT transform to an image and returns an image of the same size

        :param fft_out: original_image, fft magnitude, fft phase (output of fft2 method of FFT class)
        :return:
        """

        return self.ifft(fft_out)

    # Display the fft and the image
    def ifft(self, fft_out):
        """

        :param fft_out: original_image, fft magnitude, fft phase (output of fft2 method of FFT class)
        :return:
        """

        # 2D fourier transform
        orig_img, fft_mag, fft_phase = fft_out

        # FFT function returns magnitude in log scale, bring it back to linear
        fft_mag = 10**fft_mag

        if self.mask is not None:
            fft_mag = fft_mag * self.mask

        # Convert the magnitude and phase back into a complex number
        fft_complex = np.multiply(fft_mag, np.exp(1j * fft_phase))

        # These are the FFT images
        shift_inverted = np.fft.ifftshift(fft_complex, axes=self.axis)
        inv_img = np.real(np.fft.ifft2(shift_inverted, axes=self.axis))

        return orig_img, inv_img


class Show:
    """

    """

    def __init__(self, save_filename=None, do_show=True):
        """

        :param save_filename: Filename to save the output
        :param do_show: Display a plot or not
        """

        self.save_filename = save_filename
        self.do_show = do_show

    @staticmethod
    def _chk_shape(in_imgs):
        if len(in_imgs.shape) == 2:
            in_imgs = in_imgs[np.newaxis, :]

        return in_imgs

    def format_input(self, in_imgs):
        if isinstance(in_imgs, np.ndarray):
            in_imgs = self._chk_shape(in_imgs)
            return in_imgs
        elif isinstance(in_imgs, Iterable):
            in_imgs = [self._chk_shape(x) for x in in_imgs]
            return in_imgs
        else:
            raise TypeError('Input must be  a 2D numpy array of shape (W, H) or (N, W, H) or a list of numpy arrays')

    def __lshift__(self, in_imgs):
        """

        :param in_imgs: One numpy array or list or tuple of numpy arrays
        :return:
        """

        in_imgs = self.format_input(in_imgs)

        return self.show(in_imgs)

    def show(self, in_imgs):
        """

        :param in_imgs: List or tuple of numpy array of shape (W, H) or (N, W, H)
        :return:
        """
        import matplotlib.pyplot as plt

        if isinstance(in_imgs, np.ndarray):
            in_imgs = [in_imgs, ]

        # Number of cols and number of rows
        n_cols = len(in_imgs)
        n_rows = in_imgs[0].shape[0]
        fig = plt.figure(figsize=(6.4*n_cols, 4.8*n_rows))

        # Assumes every item on the list has the same number of dimensions
        # Walk through every image
        for row_cnt in range(in_imgs[0].shape[0]):
            # Walk through every column of the image
            for col_cnt in range(len(in_imgs)):
                img_cnt = row_cnt*n_cols + col_cnt + 1
                ax = fig.add_subplot(n_rows, n_cols, img_cnt)
                ax.imshow(np.squeeze(in_imgs[col_cnt][row_cnt, :, :]), cmap='gray')

        plt.tight_layout()
        if self.save_filename is not None:
            plt.savefig(self.save_filename)

        if self.do_show:
            plt.show()

        return in_imgs


class CreateOnesMask:
    def __init__(self, in_imgs):

        # Create an ones mask
        self.shape = in_imgs.shape[-2:]
        self.mask = np.ones(self.shape)
        self.center = np.array([int(x/2) for x in self.shape])
        self.circle_center = np.array([((x-1)/2) for x in self.shape])

    def horizontal_from_center(self, left_width, right_width, height, val=0):
        """

        :return:
        """

        top = self.center[1] - int(height / 2)
        left = self.center[0] - left_width
        width = left_width + right_width

        self.offset_box(left, top, width, height, val)

    def vertical_from_center(self, top_height, bottom_height, width, val=0):
        """

        :return:
        """

        left = self.center[0]-int(width/2)
        top = self.center[1] - top_height
        height = top_height + bottom_height

        self.offset_box(left, top, width, height, val)

    def center_box(self, width, height, val=0):
        """

        :return:
        """

        # Get the top left corner relative to center
        left = self.center[0] - int(width/2)
        top = self.center[1] - int(height/2)
        self.offset_box(left, top, width, height, val)

    def offset_box(self, left, top, width, height, val=0):
        """

        :return:
        """
        self.mask[left:left+width, top:top+height] = val

    def center_circle(self, radius, inside=True, val=0):
        """

        :param radius:
        :param inside:
        :param val:
        :return:
        """

        center_x = self.circle_center[0]
        center_y = self.circle_center[1]

        self.offset_circle(center_x, center_y, radius, inside, val)

    def offset_circle(self, center_x, center_y, radius, inside=True, val=0):
        """

        :return:
        """

        # These are all the X and Y's that make up a circle
        x, y = np.meshgrid(range(self.shape[0]), range(self.shape[1]))
        r = np.sqrt((x-center_x)**2 + (y-center_y)**2)

        # This is the radius of the circle
        if inside:
            index = r < radius
        else:
            index = r > radius

        # These are the indices to keep
        x = x[index]
        y = y[index]

        self.mask[x, y] = val


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
            raise KeyError(f'Missing required parameters: {missing_list}')

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
        
        #Check axis and create filter
        if int(axis) not in (0,1):
            raise ValueError(f'for a 2D sobel filter, axis must be equal to 0 or 1 but \'{axis}\' was provided.')
        elif int(axis) == 0:
            return np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        elif int(axis) == 1:
            return np.array([[1,0,-1],[2,0,-2],[1,0,-1]]).T
                              
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
        
        #Check axis and create filter
        if int(axis) not in (0,1):
            raise ValueError(f'for a 2D prewitt filter, axis must be equal to 0 or 1 but \'{axis}\' was provided.')
        elif int(axis) == 0:
            return np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        elif int(axis) == 1:
            return np.array([[1,0,-1],[1,0,-1],[1,0,-1]]).T

        
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

        return self.apply_filter(in_imgs, kernel)

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
            for _ in range(len(in_imgs.shape)-2):
                kernel = kernel[np.newaxis, :]

        if dim == 1:
            return in_imgs, convolve1d(in_imgs, kernel, mode=self.mode, axis=axis)
        elif dim == 2:
            # Create 2d filter and normalise
            return in_imgs, convolve(in_imgs, kernel, mode=self.mode, cval=self.cval)

class Canny:
    """
    Applies a canny filter to the image
    Wrapper for the skimage implementation
    https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html
    """

    def __init__(self, sigma, low_threshold=None, high_threshold=None, 
                 mask=None, use_quantiles=False, mode='constant', cval=0.0):
        """
        :param sigma: Standard deviation of the gaussian filter. 
        :return:  
        """
        
        self.sigma = sigma
        self.lt = low_threshold 
        self.ht = high_threshold 
        self.msk = mask 
        self.uq = use_quantiles
        self.mde = mode
        self.cval = cval       

    def __lshift__(self, in_imgs):
        """
        Applies a kernel to images and returns that image with the kernel applied.
        
        :param kern_out: Output of the Kernel class (images, kernel)
        :return:
        """
        return in_imgs, self.apply_filter(in_imgs)
            
    def apply_filter(self, in_imgs):
        """
        Applies a canny filter, essentially a wrapper for the scikit-image.feature.canny() method.
        
        To do: not vectorised. Perhaps can be implemented with joblib?
        https://scikit-image.org/docs/stable/user_guide/tutorial_parallelization.html
        
        """
        # For loop to apply the canny function to each img in the image array
        out_array = np.array([sk_canny(in_imgs[i], sigma=self.sigma, low_threshold=self.lt, high_threshold=self.ht, 
                              mask=self.msk, use_quantiles=self.uq, mode=self.mde, cval=self.cval
                                ) for i in range(len(in_imgs))])
        
        return out_array    
    

if __name__ == '__main__':

    do_kernel = True
    if do_kernel:
        n_samples = 10
        images = DefectViewer() << (ImageLoader(defect_class='FrontGridInterruption') << n_samples)

        # ck = CreateKernel(bim=2, kernel='gaussian', size=3, std=8)
        c_imgs = Convolve() << (CreateKernel(dim=2, kernel='gaussian', size=3, std=8) << images)

        # Show the original image
        Show('2dgaussian') << c_imgs

        # ck = CreateKernel(bim=2, kernel='gaussian', size=3, std=8)
        # Make first 10 rows 0
        images[:, :10, :] = 0
        c_imgs = Convolve(axis=-1) << (CreateKernel(dim=1, kernel='gaussian', size=10, std=8) << images)

        # Show the original image
        Show('1dgaussian_y') << c_imgs

        # ck = CreateKernel(bim=2, kernel='gaussian', size=3, std=8)
        c_imgs = Convolve(axis=-2) << (CreateKernel(dim=1, kernel='gaussian', size=10, std=8) << images)
        
        # Show the original image
        c_imgs = Show('1dgaussian_x') << c_imgs
        
        # test sobel filter
        c_imgs = Convolve(axis=-2) << (CreateKernel(dim=2, kernel='sobel', axis=0) << images)
        
        # Show the original image
        Show('2dgaussian') << c_imgs
        
        # test prewitt filter
        c_imgs = Convolve(axis=-2) << (CreateKernel(dim=2, kernel='prewitt', axis=0) << images)
        
        # Show the original image
        Show('2dgaussian') << c_imgs

    do_create_mask = False
    if do_create_mask:

        # Create a 10X10 image
        cm = CreateOnesMask(np.zeros((10, 10)))
        cm.horizontal_from_center(left_width=3, right_width=3, height=4)
        print(cm.mask)

        # Create a 10X10 image
        cm = CreateOnesMask(np.zeros((10, 10)))
        cm.vertical_from_center(top_height=3, bottom_height=3, width=4)
        print(cm.mask)

        # Create a 10X10 image
        cm = CreateOnesMask(np.zeros((10, 10)))
        cm.center_circle(radius=2)
        print(cm.mask)

    do_shorthand = False
    if do_shorthand:
        # Perform an FFT of an image
        # Load two samples of the defect class
        n_samples = 10
        images = DefectViewer() << (ImageLoader(defect_class='FrontGridInterruption') << n_samples)
        # FFT the images and get a tuple (original_img, magnitude and phase)
        fft_images = FFT(dim=2) << images

        # Show the original image
        fft_images = Show('fft') << fft_images

        # Recreate the original image from the FFT and display
        inv_images = IFFT(mask=np.ones(fft_images[0].shape[1:])) << fft_images

        # Display the images
        inv_images = Show('inv_fft') << inv_images

    do_test_fft = False
    if do_test_fft:
        # Load 'n' images
        imgl = ImageLoader()
        imgl.load_n(n=10, defect_classes='FrontGridInterruption')

        # defect viewer
        dv = DefectViewer(imgl)
        sample_df = dv.load_image(imgl.sample_df)

        images = sample_df['image'].tolist()
        images = [cv2.resize(x, (224, 224)) for x in images]
        images = np.stack(images, axis=0)

        # Perform an FFT of an image
        t = Show('test') << (FFT(dim=2, axis=(-2, -1)) << images) << \
            (DefectViewer() << (ImageLoader(defect_class='FrontGridInterruption') << 2))
        print('done')
