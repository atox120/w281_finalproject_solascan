import copy
import numpy as np

from scipy.signal import find_peaks

from app.imager import Show, Exposure
from app.utils import ImageWrapper, input_check
from app.filters import HOG, Convolve, CreateKernel


class Orient:
    def __init__(self, num_jobs=10, imgs_per_job=100):
        """
        :param num_jobs:
        """

        self.num_jobs = num_jobs
        self.imgs_per_job = imgs_per_job
        self.hog_params = {'pixels_per_cell': (3, 3), 'num_jobs': num_jobs}

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

    @staticmethod
    def fix_orientation(filt_hogs, in_imgs, in_hogs):
        """
        Direction of greatest orientaiton is the direction of busbars
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
            if r:
                accum_imgs.append(img.T)
                accum_hogs.append(hog.T)
            else:
                accum_imgs.append(img)
                accum_hogs.append(hog)

        out_imgs = np.stack(accum_imgs, axis=0)
        out_hogs = np.stack(accum_hogs, axis=0)

        return out_imgs, out_hogs, rotate

    def apply(self, in_imgs, do_debug=False):

        # Adaptive histogram Equalization of images
        imgs_exposure = Exposure('adaptive').apply(in_imgs)

        # Next perform a hog on the images
        hog_exposed = HOG(**self.hog_params).apply(imgs_exposure)

        # Exposure stretch the HOG image and apply the sigmoid
        hog_stretched = Exposure('stretch').apply(hog_exposed)

        # Now re-orient the images with wrong orientation
        rotated_images, rotated_hog, rotate = self.fix_orientation(hog_stretched, in_imgs, hog_exposed)
        only_rotated_images = rotated_images[rotate, :]
        only_rotated_hogs = rotated_hog[rotate, :]

        if do_debug:
            Show(num_images=10).show((imgs_exposure, hog_exposed, hog_stretched))
            Show().show((only_rotated_images, only_rotated_hogs))

        return rotated_images, rotated_hog


class BusBarMask:
    
    def __init__(self, invert=False, edge_buffer=15, filter_length=4, filter_divisor = 3, 
                 sigmoid_cutoff=0.3, hog_ratio=4,num_jobs=10, imgs_per_job=100):
        """
        Creates a mask for the busbars, along the full width of the image
        
        :param invert: True that the busbars are 1 and rest is 0
                        False for busbars are 0 and rest is 1
        :edge buffer: Distance from top and bottom edge in pixels in which we ignore any busbars
                        detected. Default is 0. 
                        
        :param filter length: 1D filter length, Increase to return thicker borders around the busbar
        :param filter_divisor: divisor on the filter, increase to reduce the value of the kernel
        """
        
        self.invert = False
        self.edge_buffer = edge_buffer
        self.filter_length = filter_length
        self.filter_divisor = filter_divisor
        
        self.hog_ratio = hog_ratio
        self.sigmoid_cutoff = sigmoid_cutoff
        
        self.num_jobs = num_jobs
        self.imgs_per_job = imgs_per_job

        self.hog_params = {'pixels_per_cell': (3, 3), 'num_jobs': num_jobs}

    def __lshift__(self, in_imw):

        # Apply the transformation to these images
        out_img = self.apply(in_imw[0].images, in_imw[1].images)
        category = in_imw[-1].category
        category += f'\n Busbar Mask'
        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw[-1].image_labels))

        return in_imw[-1], out_imw

    def get_hog_mask(self, in_hogs):
        """
        :param in_hogs:

        """
        # Get HOG transformed images
        hog_imgs = HOG(**hog_params).apply(in_hogs)

        # Exposure stretch the HOG image and apply the sigmoid
        hog_stretched = Exposure('stretch').apply(hog_imgs)

        # Mask the HOG and apply the sigmoid
        sig_hog = Exposure('sigmoid').apply(hog_stretched)

        # Sum the pixel values in the X direction
        hog_signal = sig_hog.sum(axis=-1).astype(int)
        
        #Convolve the HOG Signal
        busbar_filter = np.ones(self.filter_length) / self.filter_divisor
        [np.convolve(img, my_filter) for img in hog_signal]
        

        # These are the minimum counts per row that are acceptable
        hog_threshold = (convolved_signal.max(axis=-1) / self.hog_ratio)[:, np.newaxis]

        # All rows that are greater than the threshold for that image
        # noinspection PyUnresolvedReferences
        hog_index = (hog_signal > hog_threshold).astype(int)

        # Create row-wise mask by repeating the index along the columns
        mask = np.repeat(hog_index[:,:,np.newaxis], h_img.shape[1], axis=2)
        
        #Threshold values
        mask[mask > 0]=1

        #Apply buffer
        mask[:, 0:self.edge_buffer,:] = 0
        mask[:, -self.edge_buffer:,:] = 0


        return mask

    def apply(self, in_hogs, in_imgs, do_debug=False):
        """
        """

        # Create the HOG to apply to the images
        final_mask = self.get_hog_mask(in_hogs)

        if do_debug:
            Show(num_images=10, seed=1234).show((final_mask, in_imgs))

        return out_imgs
    
class RemoveBusBars:
    def __init__(self, num_jobs=10, imgs_per_job=100):
        """
        """

        self.num_jobs = num_jobs
        self.imgs_per_job = imgs_per_job
        self.hog_ratio = 4
        self.sigmoid_cutoff = 0.3

        self.hog_params = {'pixels_per_cell': (3, 3), 'num_jobs': num_jobs}

    def __lshift__(self, in_imw):

        # Apply the transformation to these images
        out_img = self.apply(in_imw[0].images, in_imw[1].images)

        category = in_imw[-1].category
        category += f'\n Busbar removed'
        out_imw = ImageWrapper(out_img, category=category, image_labels=copy.deepcopy(in_imw[-1].image_labels))

        return in_imw[-1], out_imw

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
        shaken_hog = sig_threshold_hog + \
            np.roll(sig_threshold_hog, shift=-1, axis=-1) + np.roll(sig_threshold_hog, shift=1, axis=-1) + \
            np.roll(sig_threshold_hog, shift=-2, axis=-1) + np.roll(sig_threshold_hog, shift=2, axis=-1) + \
            np.roll(sig_threshold_hog, shift=-3, axis=-1) + np.roll(sig_threshold_hog, shift=3, axis=-1) + \
            np.roll(sig_threshold_hog, shift=-4, axis=-1) + np.roll(sig_threshold_hog, shift=4, axis=-1)
        shaken_hog = shaken_hog + np.roll(shaken_hog, shift=1, axis=-2) + np.roll(shaken_hog, shift=-1, axis=-2)

        # Stretch the final HOG to fit 0 to 1
        final_hog = Exposure('sigmoid', cutoff=0.3).apply(Exposure('stretch').apply(shaken_hog))

        return final_hog

    def apply(self, in_hogs, in_imgs, do_debug=False):
        """
        """

        # Create the HOG to apply to the images
        final_hog = self.get_hog_mask(in_hogs)

        # Now shake the images
        out_imgs = (1 - final_hog) * in_imgs + final_hog * (
                    np.roll(in_imgs, shift=10, axis=-2) + np.roll(in_imgs, shift=-10, axis=-2)) / 2

        if do_debug:
            Show(num_images=10, seed=1234).show((final_hog, in_imgs, out_imgs))

        return out_imgs



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
            self.num_jobs = kwargs['kwargs']
            del kwargs['num_jobs']

        self.params = {}
        input_check(kwargs, 'reduce_max', 1, self.params, exception=False)
        input_check(kwargs, 'finger_mult', 10, self.params, exception=False)
        input_check(kwargs, 'finger_height', 15, self.params, exception=False)
        input_check(kwargs, 'finger_width', 8, self.params, exception=False)
        input_check(kwargs, 'padding_mult', 2, self.params, exception=False)
        input_check(kwargs, 'top_padding', 0, self.params, exception=False)
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
    def simple_finger_kernel(finger_width=2, finger_height=6, side_padding=2, top_padding=3, finger_mult=2,
                             flipped=False):
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

        if (finger_height + top_padding) % 2 == 0:
            finger_height += 1

        # Width and height of the finger
        width = finger_width + 2 * side_padding
        height = finger_height + top_padding

        #
        total_size = width * height
        finger_size = finger_width * finger_height

        # Give puter region and finger weights
        outer_weight = 1 / (total_size - finger_size)
        finger_weight = 1 / finger_size

        # Create the kernel here
        kernel = np.ones((width, height)) * outer_weight
        kernel[side_padding:side_padding + finger_width, :finger_height] = -finger_weight * finger_mult

        if kernel.shape[0] % 2 == 0:
            kernel = np.vstack((kernel, np.zeros((kernel.shape[1],))))

        kernel = kernel.T
        if flipped:
            kernel = np.flipud(kernel)

        return kernel

    @staticmethod
    def complex_finger_kernel(finger_width=2, finger_height=6, side_padding=2, top_padding=2, finger_mult=2,
                              flipped=False):
        """
                 o o f f r r
                 o o f f r r
                 o o f f r r
                 o o f f r r
                 o o f f r r
                 o o f f r r
                 o o o r r r
                 o o o r r r

        value of o = 1/(count of o)
        value of f = 1/(count of f)
        value of r = -1/(count of r)
        """

        if (finger_height + top_padding) % 2 == 0:
            top_padding += 1

        #
        width = finger_width + 2 * side_padding
        height = finger_height + top_padding

        kernel = np.zeros((width, height))

        # Side pad weight is 1/the number of elements in it
        padded_area_weight = 1 / (side_padding * height)
        # print(f'Padded area weight {padded_area_weight}')

        kernel[:side_padding, :] = padded_area_weight
        kernel[-side_padding:, :] = -padded_area_weight

        # These are the
        symmetric_elements = int(finger_width / 2)
        top_pad_weight = 1 / (symmetric_elements * top_padding)
        # print(f'Top pad weight {top_pad_weight}')

        # Setup the padding for the bottom
        kernel[side_padding:side_padding + symmetric_elements, finger_height:] = top_pad_weight
        kernel[-(side_padding + symmetric_elements):-side_padding, finger_height:] = -top_pad_weight

        # Setup the weights for the finger
        finger_weight = 1 / (finger_width * finger_height)
        kernel[side_padding:side_padding + finger_width, :finger_height] = finger_weight * finger_mult

        kernel = kernel.T
        if flipped:
            kernel = np.flipud(kernel)

        return kernel

    def apply(self, in_imgs, do_debug=False):
        """

        """

        # Form an adaptive contrast
        stretched_imgs = Exposure('adaptive').apply(in_imgs)

        all_ops = []
        after_conv = []
        after_sobel = []

        params = copy.deepcopy(self.params)
        params['side_padding'] = params['finger_width'] * params['padding_mult']

        # Convolve the image with kernel
        # Stretch the values after applying kernel
        kernel = self.simple_finger_kernel(params['finger_width'], params['finger_height'], params['side_padding'],
                                           params['top_padding'], params['finger_mult'], flipped=params['flipped'])

        conv_imgs = Convolve(num_jobs=self.num_jobs).apply(stretched_imgs, kernel)
        conv_stretch = Exposure('stretch').apply(conv_imgs)

        # Apply sobel on opt of it
        sobel_kernel = CreateKernel(kernel='sobel', axis=0).apply()
        sobel_imgs = Convolve(num_jobs=self.num_jobs).apply(conv_stretch, sobel_kernel)

        if do_debug:
            after_conv.append(conv_stretch)
            after_sobel.append(sobel_imgs)

        # All operations
        all_ops.append(sobel_imgs)

        # Take a Max across all configurations
        multi_stack = np.stack(all_ops, axis=0)

        if params['reduce_max']:
            multi_stack = np.max(multi_stack, axis=0)
        else:
            multi_stack = np.min(multi_stack, axis=0)

        # Apply HOG on the multi stack
        # hog_imgs = HOG(pixels_per_cell=(5, 5), num_jobs=20).apply(multi_stack, get_images=True)
        stretch = Exposure('stretch').apply(multi_stack)

        # Perform the HIG
        # hog_imgs = HOG(pixels_per_cell=(5, 5), num_jobs=self.num_jobs).apply(stretch, get_images=True)

        if do_debug:
            Show(num_images=10, seed=1234).show(
                (in_imgs, stretched_imgs,) + tuple(after_conv) + tuple(after_sobel) + (multi_stack, stretch))
        return stretch
