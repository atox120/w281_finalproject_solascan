import cv2
import numpy as np
from collections.abc import Iterable
from app.imager import ImageLoader, DefectViewer


class FFT:

    def __init__(self, dim=2, axis=(-2, -1), return_which='both'):
        """

        :param dim:
        :param axis: Which dimension(s) to perform FFT on. When dim is 2 then axis should be a tuple. Default is the
            last two dimensions. When dim is 1 then axis should be an integer dimension.
        :param return_which: If 'both' (default) then returns (magnitude, phase). If 'magnitude' then return only the
            magnitude. If 'phase' then return only the phase
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
        transformed = np.fft.fftshift(np.fft.fft2(in_img * win, axes=self.axis))

        if self.return_which == 'both':
            magnitude = np.log10(np.abs(transformed))
            phase = np.angle(transformed)
            return magnitude, phase
        elif self.return_which == 'magnitude':
            magnitude = np.log10(np.abs(transformed))
            return magnitude
        elif self.return_which == 'phase':
            phase = np.angle(transformed)
            return phase
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
        transformed = np.fft.fftshift(np.fft.fft(in_img * win, axis=self.axis))
        if self.return_which == 'both':
            magnitude = np.abs(transformed)
            phase = np.angle(transformed)
            return magnitude, phase
        elif self.return_which == 'magnitude':
            magnitude = np.abs(transformed)
            return magnitude
        elif self.return_which == 'phase':
            phase = np.angle(transformed)
            return phase
        else:
            raise TypeError('return_which must be one of magnitude, phase or both')


class Show:

    def __init__(self, save_filename=None, do_show=False):

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
            return TypeError('Input must be  a 2D numpy array of shape (W, H) or (N, W, H) or a list of numpy arrays')

    def __lshift__(self, in_imgs):
        """

        :param in_imgs: One numpy array or list or tuple of numpy arrays
        :return:
        """

        in_imgs = self.format_input(in_imgs)

        self.show(in_imgs)

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
                print(img_cnt)
                ax = fig.add_subplot(n_rows, n_cols, row_cnt*n_cols + col_cnt + 1)
                ax.imshow(np.squeeze(in_imgs[col_cnt][row_cnt, :, :]), cmap='gray')

        if self.save_filename is not None:
            plt.savefig(self.save_filename)

        if self.do_show:
            plt.show()

        return in_imgs


if __name__ == '__main__':
    do_test_fft = True
    if do_test_fft:
        # Load 'n' images
        imgl = ImageLoader()
        imgl.load_n(n=2)

        # defect viewer
        dv = DefectViewer(imgl)
        sample_df = dv.load_image(imgl.sample_df)

        images = sample_df['image'].tolist()
        images = [cv2.resize(x, (224, 224)) for x in images]
        images = np.stack(images, axis=0)

        # Perform an FFT of an image
        t = Show('test') << (FFT() << images)
        print('done')
