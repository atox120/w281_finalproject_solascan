import cv2
import copy
import numpy as np
from sklearn.cluster import KMeans
from app.utils import ImageWrapper, input_check


class SIFT:
    """
    https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f

    """

    def __init__(self, consume_kwargs=True, **kwargs):
        # Create a SIFt extraction object
        self.sift = cv2.xfeatures2d.SIFT_create()

        self.centers = np.zeros((1, 1))

        self.params = {}
        input_check(kwargs, 'n_clusters', 10, self.params, exception=False)
        input_check(kwargs, 'n_init', 10, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):

        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        # Apply the transformation to these images
        in_images = ~in_imw
        augmented = self._get_feature_augmented(in_images)

        category = in_imw.category
        category += f'\n SIFT'
        out_imw = ImageWrapper(augmented, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def _get_feature_augmented(self, train_imgs):
        """

        :param train_imgs: Numpy array of images
        :return:
        """

        augmented = []
        for img in train_imgs:
            # Accumulate the image descriptors and the key points
            gray = (img*255).astype('uint8')
            kp = self.sift.detect(gray, None)
            out_img = cv2.drawKeypoints(gray, kp, img)
            augmented.append(out_img[np.newaxis, :])
        augmented = np.concatenate(augmented, axis=0)
        return augmented

    def _get_descriptors(self, train_imgs):
        """

        :param train_imgs: Numpy array of images
        :return:
        """

        descriptors = []
        features = []
        for img in train_imgs:
            # Accumulate the image descriptors and the key points
            kp, descr = self.sift.detectAndCompute((img*255).astype('uint8'), None)
            if descr is None:
                descr = np.zeros((1, self.sift.descriptorSize()), np.float32)

            descriptors.extend(descr)
            features.append(descr)
        return descriptors, features

    def _find_centers(self, descriptors):
        """

        :return:
        """

        kmeans = KMeans(n_clusters=self.params['n_clusters'], n_init=self.params['n_init'])
        kmeans.fit(descriptors)

        visual_words = kmeans.cluster_centers_

        return visual_words

    def _get_histo(self, img_descriptor):
        """

        :param img_descriptor: Array of image descriptors
        :return:
        """

        # Descriptors for the
        img_descriptor = np.repeat(img_descriptor[:, np.newaxis, :], self.params['n_clusters'], axis=1)
        img_descriptor = np.sum((img_descriptor - self.centers)**2, axis=2)
        bins, counts = np.unique(np.argmin(img_descriptor, axis=1), return_counts=True)
        histo = np.zeros((self.params['n_clusters'], ))
        histo[bins] = counts

        return histo

    def fit(self, in_imgs):
        """
        Fit SIFT and find centers

        :param in_imgs: Numpy array of images
        :return:
        """

        # Find the descriptors and their centers
        descriptors, features = self._get_descriptors(in_imgs)

        # centers for the descriptors
        self.centers = self._find_centers(descriptors)
        self.centers = self.centers[np.newaxis, :, :]

        # Create histogram for each image
        histos = [self._get_histo(x) for x in features]

        return histos

    def get(self, in_imgs):
        """

        :param in_imgs:
        :return:
        """

        # Find the descriptors and their centers
        descriptors, features = self._get_descriptors(in_imgs)

        # Create histogram for each image
        histos = [self._get_histo(x) for x in features]

        return histos


class KAZE:
    """

    """

    def __init__(self, consume_kwargs=True, **kwargs):
        # Create a SIFt extraction object
        self.kaze = cv2.AKAZE_create()

        self.centers = np.zeros((1, 1))

        self.params = {}
        input_check(kwargs, 'n_clusters', 10, self.params, exception=False)
        input_check(kwargs, 'n_init', 10, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keyword(s) {kwargs.keys()}')

    def __lshift__(self, in_imw):

        if isinstance(in_imw, tuple):
            in_imw = in_imw[-1]

        # Apply the transformation to these images
        in_images = ~in_imw
        augmented = self._get_feature_augmented(in_images)

        category = in_imw.category
        category += f'\n KAZE'
        out_imw = ImageWrapper(augmented, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def _get_feature_augmented(self, train_imgs):
        """

        :param train_imgs: Numpy array of images
        :return:
        """

        augmented = []
        for img in train_imgs:
            # Accumulate the image descriptors and the key points
            gray = (img*255).astype('uint8')
            kp = self.kaze.detect(gray, None)
            out_img = cv2.drawKeypoints(gray, kp, img)
            augmented.append(out_img[np.newaxis, :])
        augmented = np.concatenate(augmented, axis=0)
        return augmented

    def _get_descriptors(self, train_imgs):
        """

        :param train_imgs: Numpy array of images
        :return:
        """

        descriptors = []
        features = []
        for img in train_imgs:
            # Accumulate the image descriptors and the key points
            kp, descr = self.kaze.detectAndCompute((img * 255).astype('uint8'), None)
            if descr is None:
                descr = np.zeros((1, self.kaze.descriptorSize()), np.float32)

            descriptors.extend(descr)
            features.append(descr)
        return descriptors, features

    def _find_centers(self, descriptors):
        """

        :return:
        """

        kmeans = KMeans(n_clusters=self.params['n_clusters'], n_init=self.params['n_init'])
        kmeans.fit(descriptors)

        visual_words = kmeans.cluster_centers_

        return visual_words

    def _get_histo(self, img_descriptor):
        """

        :param img_descriptor: Array of image descriptors
        :return:
        """

        # Descriptors for the
        img_descriptor = np.repeat(img_descriptor[:, np.newaxis, :], self.params['n_clusters'], axis=1)
        img_descriptor = np.sum((img_descriptor - self.centers)**2, axis=2)
        bins, counts = np.unique(np.argmin(img_descriptor, axis=1), return_counts=True)
        histo = np.zeros((self.params['n_clusters'], ))
        histo[bins] = counts

        return histo

    def fit(self, in_imgs):
        """
        Fit SIFT and find centers

        :param in_imgs: Numpy array of images
        :return:
        """

        # Find the descriptors and their centers
        descriptors, features = self._get_descriptors(in_imgs)

        # centers for the descriptors
        self.centers = self._find_centers(descriptors)
        self.centers = self.centers[np.newaxis, :, :]

        # Create histogram for each image
        histos = [self._get_histo(x) for x in features]

        return histos

    def get(self, in_imgs):
        """

        :param in_imgs:
        :return:
        """

        # Find the descriptors and their centers
        descriptors, features = self._get_descriptors(in_imgs)

        # Create histogram for each image
        histos = [self._get_histo(x) for x in features]

        return histos
