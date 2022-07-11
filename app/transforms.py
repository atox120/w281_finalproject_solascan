import cv2
import numpy as np
from collections.abc import Iterable
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

    def __lshift__(self, in_imgs):
        """
        Applies an FFT transform to an image and returns an image of the same size

        :param in_imgs:
        :return:
        """

        # If it is th eoutput of a different function then take the last value in the tuple
        if isinstance(in_imgs, tuple):
            in_imgs = in_imgs[-1]

        if len(in_imgs.shape) == 2:
            in_imgs = in_imgs[np.newaxis, :, :]

        # Create a window function
        win = self.create_window(in_imgs)

        if self.dim == 2:
            return self.fft2(in_imgs, win)
        else:
            return self.fft(in_imgs, win)

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
        fft_mag = 10 ** fft_mag

        if self.mask is not None:
            fft_mag = fft_mag * self.mask

        # Convert the magnitude and phase back into a complex number
        fft_complex = np.multiply(fft_mag, np.exp(1j * fft_phase))

        # These are the FFT images
        shift_inverted = np.fft.ifftshift(fft_complex, axes=self.axis)
        inv_img = np.real(np.fft.ifft2(shift_inverted, axes=self.axis))

        return orig_img, inv_img


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
