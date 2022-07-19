import copy
import numpy as np
from app.filters import HOG, Convolve
from app.imager import Show, Exposure
from app.transforms import CreateOnesMask
from app.utils import input_check, ImageWrapper


class Orient:
    def __init__(self, num_jobs=10, imgs_per_job=100):
        """
        """

        self.num_jobs = num_jobs
        self.imgs_per_job = imgs_per_job
        self.hog_boundary = 30
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
    def mask_edges(in_imgs, boundary):
        in_imgs = in_imgs.copy()

        # Create a ones mask
        cm = CreateOnesMask(in_imgs)
        cm.center_box(width=224 - 2 * boundary, height=224 - 2 * boundary)

        # Calculate the mean of the image
        mean_val = np.mean(in_imgs, axis=(-2, -1), keepdims=True)
        mask = cm.mask[np.newaxis, :]
        mask = np.ones(in_imgs.shape) * mask

        # Mask out the edges of the images
        in_imgs[mask.astype('bool')] = 0

        # Each images mean is multiplied to the mean
        mask = mask * mean_val

        # The zero areas get the mean value
        in_imgs += mask

        return in_imgs

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

        # Mask out the edges of the HOG as edges can contain wierd artifacts
        hog_edge_cleaned = self.mask_edges(hog_stretched, self.hog_boundary)

        # Now re-orient the images with wrong orientation
        rotated_images, rotated_hog, rotate = self.fix_orientation(hog_edge_cleaned, in_imgs, hog_exposed)
        only_rotated_images = rotated_images[rotate, :]
        only_rotated_hogs = rotated_hog[rotate, :]

        if do_debug:
            Show(num_images=10).show((imgs_exposure, hog_exposed, hog_stretched, hog_edge_cleaned))
            Show().show((only_rotated_images, only_rotated_hogs))

        return rotated_images, rotated_hog


class RemoveBusBars:
    def __init__(self, num_jobs=10, imgs_per_job=100):
        """
        """

        self.num_jobs = num_jobs
        self.imgs_per_job = imgs_per_job
        self.hog_boundary = 20
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

    @staticmethod
    def mask_edges(in_imgs, boundary):
        in_imgs = in_imgs.copy()

        # Create a ones mask
        cm = CreateOnesMask(in_imgs)
        cm.center_box(width=in_imgs.shape[-1] - 2*boundary, height=in_imgs.shape[-1] - 2*boundary)

        # Calculate the mean of the image
        mean_val = np.mean(in_imgs, axis=(-2, -1), keepdims=True)
        mask = cm.mask[np.newaxis, :]
        mask = np.ones(in_imgs.shape)*mask

        # Mask out the edges of the images
        in_imgs[mask.astype('bool')] = 0

        # Each images mean is multiplied to the mean
        mask = mask * mean_val

        # The zero areas get the mean value
        in_imgs += mask

        return in_imgs

    def get_hog_mask(self, in_hogs):
        """

        :param in_hogs:

        """

        # Exposure stretch the HOG image and apply the sigmoid
        hog_stretched = Exposure('stretch').apply(in_hogs)

        # Mask the HOG and apply the sigmoid
        masked_hogs = self.mask_edges(hog_stretched, boundary=self.hog_boundary)
        sig_hog = Exposure('sigmoid').apply(masked_hogs)

        # Sum the pixel values in the X direction
        hog_signal = sig_hog.sum(axis=-1).astype(int)

        # These are the minimum counts per row that are acceptable
        hog_threshold = (hog_signal.max(axis=-1) / self.hog_ratio)[:, np.newaxis]

        # All rows that are greater than the threshold for that image
        # noinspection PyUnresolvedReferences
        hog_index = (hog_signal > hog_threshold).astype(int)

        # Now apply the threshold to the image
        thresh_hog = hog_stretched * hog_index[:, :, np.newaxis]
        sig_threshold_hog = Exposure('sigmoid', cutoff=self.sigmoid_cutoff).apply(thresh_hog)

        # Shake the HOG
        shaken_hog = sig_threshold_hog + np.roll(sig_threshold_hog, shift=-1, axis=-1) + np.roll(sig_threshold_hog,
                                                                                                 shift=1, axis=-1)
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
                    np.roll(in_imgs, shift=9, axis=-2) + np.roll(in_imgs, shift=-9, axis=-2)) / 2

        if do_debug:
            Show(num_images=10, seed=1234).show((final_hog, in_imgs, out_imgs))

        return out_imgs


class HighlightFrontGrid:

    def __init__(self, **kwargs):
        """
        :Keyword arguments:
            finger_type: 'simple' or 'complex'
            finger_width: Desired width of the finger
            finger_height: height of the finger
            side_padding: Width of outer region of finger
            top_padding: Amount of padding on the top of the finger
            flipped: Finger pointing up or down(default)

        """

        self.finger_type = kwargs['finger_type']
        del [kwargs['finger_type']]
        self.params = {}

        input_check(kwargs, 'finger_width', 2, self.params, exception=False)
        input_check(kwargs, 'finger_height', 6, self.params, exception=False)
        input_check(kwargs, 'side_padding', 2, self.params, exception=False)
        input_check(kwargs, 'top_padding', 3, self.params, exception=False)
        input_check(kwargs, 'flipped', False, self.params, exception=False)

        if kwargs:
            raise KeyError(f'Unused keys in kwargs {kwargs.keys()}')

    @staticmethod
    def simple_finger_kernel(finger_width=2, finger_height=6, side_padding=2, top_padding=3, flipped=False):
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
            top_padding += 1

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
        kernel[side_padding:side_padding + finger_width, :finger_height] = -finger_weight

        if kernel.shape[0] % 2 == 0:
            kernel = np.vstack((kernel, np.zeros((kernel.shape[1],))))

        kernel = kernel.T
        if flipped:
            kernel = np.flipud(kernel)

        return kernel

    @staticmethod
    def complex_finger_kernel(finger_width=2, finger_height=6, side_padding=2, top_padding=2, flipped=False):
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
        kernel[side_padding:side_padding + finger_width, :finger_height] = finger_weight

        kernel = kernel.T
        if flipped:
            kernel = np.flipud(kernel)

        return kernel

    def apply(self, in_imgs, do_debug=False):
        """

        """

        if self.finger_type == 'simple':
            kernel = self.simple_finger_kernel(**self.params)
        elif self.finger_type == 'complex':
            kernel = self.complex_finger_kernel(**self.params)
        else:
            raise KeyError(f'finger_type can only be simple or complex')

        #
        out_imgs = Convolve().apply(in_imgs, kernel)

        if do_debug:
            Show(num_images=10, seed=1234).show((in_imgs, out_imgs))

        return out_imgs
