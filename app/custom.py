import copy
import numpy as np
# noinspection PyUnresolvedReferences
from scipy.signal import find_peaks, gaussian
from app.imager import Show, Exposure
from app.filters import HOG, Convolve, Farid
from app.utils import ImageWrapper, input_check, line_split_string


class Orient:
    def __init__(self, num_jobs=10, imgs_per_job=100, do_debug=False, do_eliminate=False):
        """
        :param num_jobs:
        """

        self.num_jobs = num_jobs
        self.imgs_per_job = imgs_per_job
        self.hog_params = {'pixels_per_cell': (3, 3), 'num_jobs': num_jobs}
        self.border = 20
        self.do_debug = do_debug
        self.do_eliminate = do_eliminate

    def __lshift__(self, in_imw):

        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        # Apply the transformation to these images
        out_tuple = self.apply(in_imw.images)

        category = in_imw.category
        category += f'\n re-oriented hogs'
        out_imw_0 = ImageWrapper(out_tuple[1], category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        category = in_imw.category
        category += f'\n re-oriented images'
        out_imw_1 = ImageWrapper(out_tuple[0], category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return out_imw_0, out_imw_1

    def fix_orientation(self, filt_hogs, in_imgs, in_hogs):
        """
        Direction of greatest orientation is the direction of busbars
        :param filt_hogs:
        :param in_hogs:
        :param in_imgs:
        """

        # Find the gradients in the x-direction
        sum_gradients_x = np.sum(filt_hogs, axis=-1, keepdims=False)
        max_gradients_x = np.max(sum_gradients_x, axis=-1)

        # Find the gradients in the y-direction
        sum_gradients_y = np.sum(filt_hogs, axis=-2, keepdims=False)
        max_gradients_y = np.max(sum_gradients_y, axis=-1)

        # These are the images that require rotation
        accum_imgs = []
        accum_hogs = []
        rotate = max_gradients_y > max_gradients_x
        for cnt, r in enumerate(rotate):
            img = np.squeeze(in_imgs[cnt, :, :])
            hog = np.squeeze(in_hogs[cnt, :, :])
            if r and not self.do_eliminate:
                accum_imgs.append(img.T)
                accum_hogs.append(hog.T)
            else:
                accum_imgs.append(img)
                accum_hogs.append(hog)

        out_imgs = np.stack(accum_imgs, axis=0)
        out_hogs = np.stack(accum_hogs, axis=0)

        return out_imgs, out_hogs, rotate

    def apply(self, in_imgs, do_debug=False):

        do_debug |= self.do_debug

        # Adaptive histogram Equalization of images
        imgs_exposure = Exposure('adaptive').apply(in_imgs)

        # Next perform a hog on the images
        hog_exposed = HOG(**self.hog_params).apply(imgs_exposure)

        # Exposure stretch the HOG image and apply the sigmoid
        b = self.border
        hog_stretched = Exposure('stretch').apply(hog_exposed)
        # Remove the edges when calculating the HOG
        hog_stretched = hog_stretched[:, b:-b, b:-b]

        # Now re-orient the images with wrong orientation
        rotated_images, rotated_hog, rotate = self.fix_orientation(hog_stretched, in_imgs, hog_exposed)
        only_rotated_images = rotated_images[rotate, :]
        only_rotated_hogs = rotated_hog[rotate, :]
        rotate_before_images = in_imgs[rotate, :]
        hog_before_rotation = hog_stretched[rotate, :]

        if do_debug:
            # Show(num_images=100).show((imgs_exposure, hog_exposed, hog_stretched))
            Show().show((rotate_before_images, hog_before_rotation, only_rotated_hogs, only_rotated_images))

        return rotated_images, rotated_hog


class RemoveBusBars:
    def __init__(self, consume_kwargs=True, **kwargs):
        """
        """

        if 'num_jobs' in kwargs:
            self.num_jobs = kwargs['num_jobs']
            del kwargs['num_jobs']
        else:
            self.num_jobs = 1

        if 'imgs_per_job' in kwargs:
            self.imgs_per_job = kwargs['imgs_per_job']
            del kwargs['imgs_per_job']
        else:
            self.imgs_per_job = 100

        if 'sigmoid_cutoff' in kwargs:
            self.sigmoid_cutoff = kwargs['sigmoid_cutoff']
            del kwargs['sigmoid_cutoff']
        else:
            self.sigmoid_cutoff = 0.3

        if 'hog_ratio' in kwargs:
            self.hog_ratio = kwargs['hog_ratio']
            del kwargs['hog_ratio']
        else:
            self.hog_ratio = 4

        if 'replace_type' in kwargs:
            self.replace_type = kwargs['replace_type']
            del kwargs['replace_type']
        else:
            self.replace_type = 'copy'

        if 'keep_bars' in kwargs:
            self.keep_bars = kwargs['keep_bars']
            del kwargs['keep_bars']
        else:
            self.keep_bars = False

        self.params = {}
        input_check(kwargs, 'pixels_per_cell', (3, 3), self.params, exception=False)
        input_check(kwargs, 'cells_per_block', (3, 3), self.params, exception=False)
        input_check(kwargs, 'block_norm', 'L2-Hys', self.params, exception=False)
        input_check(kwargs, 'transform_sqrt', False, self.params, exception=False)
        input_check(kwargs, 'feature_vector', True, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):

        # Apply the transformation to these images
        if len(in_imw) == 2:
            # Both HOG and regular images are provided
            hog_images = in_imw[0].images
            images = in_imw[1].images
        else:
            # Only regular images are provided
            hog_images = None
            images = in_imw[-1].images

        # Apply BusBar removal
        out_img = self.apply(images, hog_images)

        category = in_imw[-1].category
        category += f'\n Busbar removed'
        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw[-1].image_labels))

        return in_imw[-1], out_imw

    def do_hog(self, in_imgs):
        # Adaptive histogram Equalization of images
        imgs_exposure = Exposure('adaptive').apply(in_imgs)

        # Next perform a hog on the images
        params = self.params
        params.update({'num_jobs': self.num_jobs})
        hog_exposed = HOG(**params).apply(imgs_exposure)

        # Exposure stretch the HOG image and apply the sigmoid
        hog_stretched = Exposure('stretch').apply(hog_exposed)

        return hog_stretched

    def get_hog_mask(self, in_hogs):
        """
        :param in_hogs:
        """

        # Exposure stretch the HOG image and apply the sigmoid
        hog_stretched = Exposure('stretch').apply(in_hogs)

        # Mask the HOG and apply the sigmoid
        sig_hog = Exposure('sigmoid').apply(hog_stretched)

        # Sum the pixel values in the X direction
        hog_signal = sig_hog.sum(axis=-1).astype(int)

        # These are the minimum counts per row that are acceptable
        hog_threshold = (hog_signal.max(axis=-1) / self.hog_ratio)[:, np.newaxis]

        # All rows that are greater than the threshold for that image
        # noinspection PyUnresolvedReferences
        hog_index = (hog_signal > hog_threshold).astype(int)

        # Now apply the threshold to the image
        thresh_hog = hog_stretched * hog_index[:, :, np.newaxis]
        sig_threshold_hog = Exposure(mode='sigmoid', cutoff=self.sigmoid_cutoff).apply(thresh_hog)

        # Shake the HOG
        shaken_hog = sig_threshold_hog
        for shift in range(1, 3):
            shaken_hog += np.roll(sig_threshold_hog, shift=-shift, axis=-1) + \
                          np.roll(sig_threshold_hog, shift=shift, axis=-1)
        shaken_hog = shaken_hog + np.roll(shaken_hog, shift=1, axis=-2) + np.roll(shaken_hog, shift=-1, axis=-2)

        # Stretch the final HOG to fit 0 to 1
        final_hog = Exposure('sigmoid', cutoff=0.3).apply(Exposure('stretch').apply(shaken_hog))

        return final_hog

    def apply(self, in_imgs, in_hogs=None, do_debug=False):
        """
        """

        # If HOG images are not provided then do HOG
        if in_hogs is None:
            in_hogs = self.do_hog(in_imgs)

        # Create the HOG to apply to the images
        final_hog = self.get_hog_mask(in_hogs)

        # Now shake the images
        if self.replace_type == 'copy':
            out_imgs = (1 - final_hog) * in_imgs + final_hog * (
                    np.roll(in_imgs, shift=10, axis=-2) + np.roll(in_imgs, shift=-10, axis=-2)) / 2
        elif self.replace_type == 'average':
            avg = ((np.roll(in_imgs, shift=10, axis=-2) + np.roll(in_imgs, shift=-10, axis=-2))/2).mean(axis=(-2, -1))
            avg = avg[:, np.newaxis, np.newaxis]

            out_imgs = (1 - final_hog) * in_imgs + final_hog * avg
        else:
            if self.keep_bars:
                out_imgs = final_hog * in_imgs
            else:
                out_imgs = (1 - final_hog) * in_imgs

        if do_debug:
            Show(num_images=10, seed=1234).show((final_hog, in_imgs, out_imgs))

        return out_imgs


class BusbarMask:

    def __init__(self, consume_kwargs=True, **kwargs):
        """
        Creates a mask for the busbars, along the full width of the image
        
        :param edge_buffer: set these top and bottom pixels zero - ignore any features here. 
        :param min_spacing: min spacing between busbars
        :param blur_width: the width of the blurring kernel on the farid filtered signal
        :param broadening: the width around the estimated centre of the busbar which is broadened
        """

        if 'edge_buffer' in kwargs:
            self.edge_buffer = kwargs['edge_buffer']
            del kwargs['edge_buffer']
        else:
            self.edge_buffer = 15

        if 'min_spacing' in kwargs:
            self.min_spacing = kwargs['min_spacing']
            del kwargs['min_spacing']
        else:
            self.min_spacing = 30

        if 'blur_width' in kwargs:
            self.blur_width = kwargs['blur_width']
            del kwargs['blur_width']
        else:
            self.blur_width = 5

        if 'blur_sigma' in kwargs:
            self.blur_sigma = kwargs['blur_sigma']
            del kwargs['blur_sigma']
        else:
            self.blur_sigma = 1.5

        if 'broadening' in kwargs:
            self.broadening = kwargs['broadening']
            del kwargs['broadening']
        else:
            self.broadening = 3

        self.params = {}
        self.debug = False

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):

        if isinstance(in_imw, tuple):
            transformed = in_imw[-1]
            in_imw = in_imw[0]
        else:
            transformed = in_imw

        out_img = (1 - self.apply(in_imw.images)) * transformed.images

        # If it is the output of a different function then take the last value in the tuple
        category = f'\n Busbar Masked'
        if self.params:
            category += f' and params: {self.params}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs):
        """
        Wrapper to apply function, standardises syntax. 
        Also a means to access the mask directly.
        """
        masks = self.get_mask(in_imgs)

        return masks

    def get_mask(self, in_imgs):

        # Apply Farid Horizontal filter to find gradient
        filter_imgs = Farid(how='horizontal').apply(in_imgs)
        row_signal = filter_imgs.sum(axis=-1).astype(int) ** 2

        # Find centre points of busbars
        filtered_imgs, found_peaks = self.filter_1d(row_signal)

        # Now we need to create the mask broadened around the peak
        n = filtered_imgs.shape[1]  # Length of the vector
        indx_list = []
        for i in range(filtered_imgs.shape[0]):
            # Broaden the peaks
            broadened_vec = self.broaden_peak(found_peaks[i], self.broadening)
            # 1 hot encoding of busbar positions
            indx = np.zeros(n)
            indx[broadened_vec] = 1
            indx_list.append(indx)
        # Stack back together
        peak_vec = np.stack(indx_list, axis=0)

        # Create mask by stretching out the mask to fill out the columns
        mask = np.repeat(peak_vec[:, :, np.newaxis], peak_vec.shape[1], axis=2)

        if self.debug:
            return mask, peak_vec, filtered_imgs, found_peaks, row_signal
        else:
            return mask

    def filter_1d(self, signal):
        """
        Filters a 1D signal by:
        - cleaning an edge buffer to zero.
        - blurring the 1D signal to merge peaks
        """

        if self.edge_buffer > 0:
            # Fitler top and bottom
            signal[:, 0:self.edge_buffer] = 0
            signal[:, -self.edge_buffer:] = 0

        # Convolve to blur peak
        kernel = self.create_1d_gaussian(self.blur_width, self.blur_sigma)
        out_list = []
        for sig in signal:
            convolved = np.convolve(sig, kernel, mode='same')
            out_list.append(convolved)
        convolved_signals = np.stack(out_list, axis=0)

        peaks = []
        # Find the peaks
        for sig in convolved_signals:
            found_peaks, _ = find_peaks(sig, distance=self.min_spacing)
            peaks.append(found_peaks)

        return convolved_signals, peaks

    @staticmethod
    def create_1d_gaussian(size, sigma):
        """ 
        create a 2-D gaussian blurr filter for a given size and sigma 
        """
        return gaussian(size, sigma)

    @staticmethod
    def broaden_peak(signal, width):
        """
        Boarden the peaks in the signal by width on either side of the centre position. 
        """
        row_coords = []
        for i in range(-width, width + 1, 1):
            [row_coords.append(x) for x in signal + i]

        return row_coords


class HighlightFrontGrid:

    def __init__(self, consume_kwargs=True, **kwargs):
        """
        :param consume_kwargs: If True then check for unused kwargs
        :param finger_mult: How much to weight inside of finger as compared to the outside
        :param num_jobs: Number of jobs to parallelize HOG
        :param reduce_max: Reduce using max(1) or min(0)
        :param max_finger_width: Maximum finger width
        :param finger_height: Height of finger
        """

        if 'num_jobs' in kwargs:
            self.num_jobs = kwargs['num_jobs']
            del kwargs['num_jobs']
        else:
            self.num_jobs = 1

        self.params = {}
        input_check(kwargs, 'finger_mult', 1, self.params, exception=False)
        input_check(kwargs, 'finger_height', 10, self.params, exception=False)
        input_check(kwargs, 'finger_width', 8, self.params, exception=False)
        input_check(kwargs, 'side_padding', 2, self.params, exception=False)
        input_check(kwargs, 'top_padding', 0, self.params, exception=False)
        input_check(kwargs, 'bottom_padding', 0, self.params, exception=False)
        input_check(kwargs, 'flipped', False, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):

        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        # Apply the transformation to these images
        out_img = self.apply(~in_imw)

        category = in_imw.category
        category += f'\n FrontGrid'
        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    @staticmethod
    def simple_finger_kernel(finger_width=2, finger_height=6, side_padding=2, top_padding=3, bottom_padding=3,
                             finger_mult=2, flipped=False):
        """
        Kernel of a shape that highlights a finger
                 o o f f o o
                 o o f f o o
                 o o f f o o
                 o o f f o o
                 o o f f o o
                 o o f f o o
                 o o o o o o
                 o o o o o o
        value of o = 1/(count of o)
        value of f = -1/(count of f)
        """

        if (bottom_padding + finger_height + top_padding) % 2 == 0:
            finger_height += 1

        # Width and height of the finger
        width = finger_width + 2 * side_padding
        height = finger_height + top_padding + bottom_padding

        #
        total_size = width * height
        finger_size = finger_width * finger_height

        # Give puter region and finger weights
        outer_weight = 1 / (total_size - finger_size)
        finger_weight = 1 / finger_size

        # Create the kernel here
        kernel = np.ones((width, height)) * outer_weight
        kernel[side_padding:side_padding + finger_width, bottom_padding:bottom_padding + finger_height] = \
            -finger_weight * finger_mult

        if kernel.shape[0] % 2 == 0:
            kernel = np.vstack((kernel, np.zeros((kernel.shape[1],))))

        kernel = kernel.T
        if flipped:
            kernel = np.flipud(kernel)

        return kernel

    def apply(self, in_imgs):
        """

        :param: in_imgs:
        :return:

        """

        #  Remove the bus bars from the data
        nobus_images = RemoveBusBars(num_jobs=self.num_jobs, replace_type='average').apply(in_imgs)

        # Make the exposire daptive to highlight the bars
        exposure = Exposure('adaptive').apply(nobus_images)

        # This is the kernel on the image
        kernel = self.simple_finger_kernel(**self.params)

        # Apply the kernel as is
        front = Convolve(axis=1).apply(exposure, kernel)

        # Reverse the sign of the kernel and run it
        front_flipped = Convolve(axis=1).apply(exposure, kernel=-1 * kernel)

        # Take a delta of kernel run both ways
        delta = front - front_flipped

        # Stretch to make it fit 0 to 1
        stretched = Exposure('stretch').apply(delta)

        return stretched
