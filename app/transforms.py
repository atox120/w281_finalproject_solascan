import cv2
import copy
import numpy as np
from collections.abc import Iterable
from sklearn.decomposition import PCA as SKPCA
from app.imager import ImageLoader, DefectViewer, Show
from app.utils import input_check, ImageWrapper, line_split_string


class FFT:

    def __init__(self, dim=2, axis=(-2, -1)):
        """

         :param dim:
         :param axis: Which dimension(s) to perform FFT on.
                When dim is 2 then axis should be a tuple. Default is the last two dimensions.
                When dim is 1 then axis should be an integer dimension.
        """

        self.dim = dim
        self.axis = axis
        if self.dim == 1:
            if isinstance(axis, Iterable):
                raise TypeError('A single dimension FFT can only be done on one axis')

    def __lshift__(self, in_imw):
        """

        :param in_imw:
        :return:
        """

        if isinstance(in_imw, Iterable):
            # noinspection PyUnresolvedReferences
            in_imw = in_imw[-1]

        # These are the images we want to process
        in_imgs = in_imw.images
        out_tuples = self.apply(in_imgs)

        # First for the magnitude
        category = in_imw.category
        category += f'\n FFT amplitude with Hanning window'
        out_imw_0 = ImageWrapper(out_tuples[0], category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        category = in_imw.category
        category += f'\n FFT phase with Hanning window'
        out_imw_1 = ImageWrapper(out_tuples[1], category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw_0, out_imw_1

    def apply(self, in_imgs):
        # Create a window function
        win = self.create_window(in_imgs)

        if self.dim == 2:
            out_tuples = self.fft2(in_imgs, win)
        else:
            out_tuples = self.fft(in_imgs, win)

        return out_tuples

    @staticmethod
    def create_window(in_img):
        # Create a window function
        win = np.outer(np.hanning(in_img.shape[-2]), np.hanning(in_img.shape[-1]))
        win = win / np.mean(win)

        # Make the dimensions such that it works for the element wise multiplication
        for _ in range(len(in_img.shape) - 2):
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

        magnitude = np.log10(np.abs(transformed))
        phase = np.angle(transformed)
        return magnitude, phase

    # Display the fft and the image
    def fft(self, in_img, win):
        """

        :param in_img: Grey scale images to be transformed of the shape (N, height, width)
        :param win: Windowing function array
        :return:
        """

        # 2D fourier transform
        transformed = np.fft.fftshift(np.fft.fft(in_img * win, axis=self.axis), axes=self.axis)

        magnitude = np.log10(np.abs(transformed))
        phase = np.angle(transformed)
        return magnitude, phase


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

        fft_array = [x.images for x in fft_out]
        out_img = self.apply(fft_array)

        # This is the category of the original image
        category = fft_out[0].category
        category += '\n IFFT'
        if self.mask is not None:
            category += f' with mask'

        ifft_out = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(fft_out[0].image_labels))

        return fft_out[0], ifft_out

    # Display the fft and the image
    def apply(self, fft_out):
        """

        :param fft_out: original_image, fft magnitude, fft phase (output of fft2 method of FFT class)
        :return:
        """

        # 2D fourier transform
        orig_img, fft_mag, fft_phase = fft_out

        # FFT function returns magnitude in log scale, bring it back to linear
        fft_mag = 10 ** fft_mag

        if self.mask is not None:
            fft_mag = fft_mag * self.mask

        # Convert the magnitude and phase back into a complex number
        fft_complex = np.multiply(fft_mag, np.exp(1j * fft_phase))

        # These are the FFT images
        shift_inverted = np.fft.ifftshift(fft_complex, axes=self.axis)
        inv_img = np.real(np.fft.ifft2(shift_inverted, axes=self.axis))

        return inv_img


class CreateOnesMask:
    def __init__(self, in_imgs):

        # Create an ones mask
        self.shape = in_imgs.shape[-2:]
        self.mask = np.ones(self.shape)
        self.center = np.array([int(x / 2) for x in self.shape])
        self.circle_center = np.array([((x - 1) / 2) for x in self.shape])

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

        left = self.center[0] - int(width / 2)
        top = self.center[1] - top_height
        height = top_height + bottom_height

        self.offset_box(left, top, width, height, val)

    def center_box(self, width, height, val=0):
        """

        :return:
        """

        # Get the top left corner relative to center
        left = self.center[0] - int(width / 2)
        top = self.center[1] - int(height / 2)
        self.offset_box(left, top, width, height, val)

    def offset_box(self, left, top, width, height, val=0):
        """

        :return:
        """
        self.mask[left:left + width, top:top + height] = val

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
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # This is the radius of the circle
        if inside:
            index = r < radius
        else:
            index = r > radius

        # These are the indices to keep
        x = x[index]
        y = y[index]

        self.mask[x, y] = val


class PCA:
    """
    Performs Principal component analysis on either a single image or a collection
    of N images. Uses the fit_transform(X) method, Returns the inverse transform to 
    transform the data back to it's original space. 
    
    Wrapper for the sklearn implenetation
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    """

    def __init__(self, transpose=True, **kwargs):
        """
        :transpose: Whether to vectorise the implementation or not. 
            if True, the image list is transformed from N x H x W (i.e. 3D)
            to N x (HxW) which enables a single function call.  
            if False, a the PCA is performed on each individual image in a 
            for loop. 
            Note when True the PCA is conducted across all images, whereas when 
            False the PCA is conducted on a single image. 
        
        :param n_components: number of components to keep. 
        :param copy: creates a new copy, default is True.

        :return:
        """

        self.transpose = transpose
        self.params = {}

        input_check(kwargs, 'n_components', None, self.params, exception=True)
        input_check(kwargs, 'copy', True, self.params, exception=False)
        input_check(kwargs, 'whiten', False, self.params, exception=False)
        input_check(kwargs, 'svd_solver', 'auto', self.params, exception=False)
        input_check(kwargs, 'tol', 0, self.params, exception=False)
        input_check(kwargs, 'iterated_power', 'auto', self.params, exception=False)
        if self.params['svd_solver'] == 'randomized':
            input_check(kwargs, 'n_oversamples', 10, self.params, exception=False)
            input_check(kwargs, 'power_iteration_normalizer', 'auto', self.params, exception=False)
            input_check(kwargs, 'random_state', None, self.params, exception=False)

        if kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):
        """

        :param in_imw:
        :return:
        """

        if isinstance(in_imw, Iterable):
            # noinspection PyUnresolvedReferences
            in_imw = in_imw[-1]

        # These are the images we want to process
        in_imgs = in_imw.images
        out_imgs = self.apply(in_imgs)

        # First for the magnitude
        category = f'PCA with params {self.params}'
        category = in_imw.category + line_split_string(category)
        out_imw = ImageWrapper(out_imgs, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def pca_transform(self, in_imgs):
        """
        Performs the dimensionality reduction, and then returns the image
        to the original space. 
        """
        pca_ = SKPCA(**self.params)
        x_new = pca_.fit_transform(in_imgs)
        x_out = pca_.inverse_transform(x_new)

        return x_out

    def apply(self, in_imgs):
        """
        If the transpose method is specified, it transforms the image and applies the
        transform by calling the pca_transform() function. Else, we perform the 
        pca_transform() in a loop for each image. 

        """
        if self.transpose:
            # get dimensions and reshape from (N, H, W) to (N, H*W)
            n, h, w = in_imgs.shape
            new_matrix = in_imgs.reshape(n, h * w).T

            # Call function and reshape back to (N, H, W) 
            out_matrix = self.pca_transform(new_matrix)
            out_imgs = out_matrix.T.reshape(n, h, w)

        else:
            out_list = [self.pca_transform(x) for x in in_imgs]
            out_imgs = np.stack(out_list, axis=0)

        return out_imgs


if __name__ == '__main__':

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

    do_shorthand = True
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
        inv_images = IFFT(mask=None) << fft_images

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
