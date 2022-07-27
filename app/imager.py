import os
import random
import cv2
import json
import copy
import numpy as np
import pandas as pd
from skimage import exposure
import matplotlib.pyplot as plt
from random import shuffle as shuf
from collections.abc import Iterable
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from app.utils import input_check, ImageWrapper, chunk, line_split_string


class ImageLoader:
    """
    Easy loading, splitting into test and CV, and viewing of images


    """

    def __init__(self, shuffle=True, defect_class=None, is_not=False):
        """

        :param shuffle: Shuffle the samples before
        :param defect_class: None -> all defect classes. List -> ['FrontGridInterruption', 'Closed'], just these
            classes. string -> 'FrontGridInterruption', one class
        """
        
        self.shuffle = shuffle
        self.defect_class = defect_class
        self.is_not = is_not

        # Where the processed annotations csv file is stored.
        # if running windows, we need to change the backslashes to forward slashes. 
        if os.name == 'nt':
            this_file_location = os.path.abspath("").replace("\\", "/")
        else:
            this_file_location = os.path.abspath("")

        # This is the
        self.train_file_loc = os.path.join(this_file_location, "../data/images/train")
        self.test_file_loc = os.path.join(this_file_location, "../data/images/test")

        # These are the hand selected good files
        all_good = pd.read_csv("../data/andi_segmented.csv")
        all_bad = all_good[np.logical_not(all_good['clean'])]

        # Keep only FrontGrid from this file
        front_grid_df = pd.read_csv("../data/front_grid.csv")

        # Load the multiple DataFrames
        self.annotations_df = pd.read_csv("../data/processed_annotations.csv")
        self.train_files_df = pd.read_csv("../data/train.csv")
        self.test_files_df = pd.read_csv("../data/test.csv")

        # Remove all bad data files from annotations
        self.annotations_df = \
            self.annotations_df[np.logical_not(self.annotations_df['filename'].isin(all_bad['filename']))]

        # Remove all Front Grid Interruptions
        self.annotations_df = self.annotations_df[self.annotations_df['defect_class'] != 'FrontGridInterruption']
        # Add the hand selected ones back
        self.annotations_df = pd.concat((self.annotations_df, front_grid_df[self.annotations_df.columns]))
        
        ## Clean up the Cracks defect class 
        file_list = ['../data/Closed_.csv', '../data/Resistive_.csv', '../data/Isolated_.csv']
        defect_list = ['Closed', 'Resistive', 'Isolated']
        original_cols = self.annotations_df.keys()

        for i in range(len(file_list)):

            #Load corrections
            corrections = pd.read_csv(file_list[i])
            # Merge onto lain df
            image_combined = self.annotations_df.merge(corrections, left_on='filename', right_on='filename', how='left')
            # process the corrections
            image_combined['indx'] =image_combined.apply(
                lambda x: self.convert_annotations(x.clean, x.defect_class, f'{defect_list[i]}'), axis=1)
            # Apply filter
            corrected_df = image_combined[image_combined['indx']]
            # Revert to original columns
            self.annotations_df = corrected_df[original_cols]

        # Keep only one copy of the file and folder
        self.cv_files_df = pd.DataFrame()
        self.sample_df = pd.DataFrame()

        # Keep only rows with train data
        self.main_df = self.annotations_df.merge(self.train_files_df, on='filename', how='inner')
        self.test_df = self.annotations_df.merge(self.test_files_df, on='filename', how='inner')

        self.main_df['sno'] = list(range(0, self.main_df.shape[0]))

        # List of all defect classes
        self.defect_classes = self.annotations_df['defect_class'].unique().tolist()

        # Count of each class in complete dataset.
        self.instance_count = dict(self.annotations_df['defect_class'].value_counts())

    def __lshift__(self, n):

        return self.load_n(n, shuffle=self.shuffle, defect_classes=self.defect_class, is_not=self.is_not)
    
    def convert_annotations(self, x, y, filter_string):
        """converts type of annotation from the annotation checker into boolean
        :param x: the annotation as string. 
        :param y: the defect class to which this is applied
        :param filter_string: filter value
        :return: Boolean
        """

        if y == filter_string:
            if (x == 'TRUE') | (x == True) | (x == 'True'):
                return True
            elif (x == 'FALSE') | (x == 'spider') | (x == False) | (x == 'False'):
                return False
            else:
                return f"Debug {x} {y}"
        else:
            return True

    def split_train_cv(self,  train_split=0.8, seed=2**17):
        """
        Split train set in

        :param train_split:
        :param seed:
        :return:
        """

        train_and_cv_files_df = self.train_files_df.copy()

        # Set the seed for splitting train and CV
        np.random.seed(seed)
        unique_files = train_and_cv_files_df['filename'].unique()
        np.random.shuffle(unique_files)

        # These are the train and test sets
        train_set = unique_files[:int(unique_files.shape[0] * train_split)]
        cv_set = unique_files[int(unique_files.shape[0] * train_split):]

        # Split the train data
        # 1. Into train set
        self.train_files_df = train_and_cv_files_df[train_and_cv_files_df['filename'].isin(train_set)]
        # 2. Into CV set
        self.cv_files_df = train_and_cv_files_df[train_and_cv_files_df['filename'].isin(cv_set)]

    def n_instances(self, defect_class):
        # returns the number of instances in a given defect class

        return self.instance_count[defect_class]

    def load_n(self, n, shuffle=True, defect_classes=None, is_not=False):
        """
        Loads n images and returns a DataFrame with following schema:
            [filename, bounding_boxes, annotation_shape, segmentations]
            filename -> source image file.
            bounding_boxes -> extracted bounding box from the original segmentations
            annotation_shape -> the original segmentation shape (rect, circle, elipse, polygon).
            segmentations -> the original segmentation coordinates.

        :param n: Number of images to load
        :param shuffle: If True then random images are picked
        :param defect_classes: If None, then all defect classes are returned.
                             If a string, then that defect class is extracted
                             If a list, then all the defect classes in the list are extracted
        :return: DataFrame with samples

        Usage:


        """ 
        if defect_classes is not None:
            # If it is not a list then make it a list
            if not isinstance(defect_classes, list):
                if isinstance(defect_classes, str):
                    # Make it a list
                    defect_classes = [defect_classes, ]
                else:
                    raise ValueError('defect_classes can only be one of string(defect class name), '
                                     'list of string of defect classes or None(all classes)')
            if not set(defect_classes).issubset(set(self.defect_classes)):
                raise ValueError(f'Defect classes can only be one of {self.defect_classes}')
        else:
            defect_classes = copy.copy(self.defect_classes)

        # Shuffle if required
        if shuffle:
            self.main_df = self.main_df.sample(frac=1)
        else:
            self.main_df = self.main_df.sort_values(by='sno')

        # Return the DataFrame with the classes and the images
        # self.sample_df = self.main_df.groupby('defect_class').head(n) - #AT I think this shold be the last step.  
        self.sample_df = self.main_df[['filename', 'bounding_box_coords',
                                         'annotation_shape', 'region_shape_attributes',
                                         'defect_class']]
        
        # Process whether we are selecting the defect class (is_not = False) 
        # or all other classes except the defect class (is_not = True)
        if self.is_not:
            self.sample_df = self.sample_df[np.logical_not(self.sample_df['defect_class'].isin(defect_classes))]
        else:
            # If defect classes were provided then only keep the required ones
            self.sample_df = self.sample_df[self.sample_df['defect_class'].isin(defect_classes)]

        #Filter to get n instances. 
        self.sample_df = self.sample_df.head(n) 

        # Add the location of the files to read from
        self.sample_df['fileloc'] = [os.path.join(self.train_file_loc, x) for x in self.sample_df['filename']]

        return self.sample_df


class DefectViewer:
    """
    Visualizes defects and provides annotations in the form of bounding boxes or segmentations
    """

    def __init__(self, il_obj=None, resize_shape=(224, 224), row_chop=0, col_chop=0):
        """
        :param il_obj: Object of ImageLoader. Default(None)
        :param resize_shape: Shape to resize the images when Default(224, 224)
        :param row_chop: How many rows to remove from top and bottom
        :param col_chop: How many cols to remove from left and right
        """

        self.il_obj = il_obj
        self.resize_shape = resize_shape
        self.row_chop = row_chop
        self.col_chop = col_chop

    def __lshift__(self, sample_df):

        return ImageWrapper(self.shift_image_load(sample_df, self.resize_shape, (self.row_chop, self.col_chop)),
                            image_labels=sample_df['filename'].tolist(),
                            category='original', )

    @staticmethod
    def shift_image_load(in_df_filename_or_list, resize_shape=(224, 224), chop=(0, 0)):
        """
        Reads the images into numpy arrays from the source
        :param in_df_filename_or_list: if DataFrame then reads the fileloc colum of the DataFrame and creates an
            image column. If string then assumes that is a filename(with path) and returns numpy array. If list then
            assumes a list of filenames(with path) and returns a list of numpy arrays
        :param resize_shape:
        :param chop:
        :return: Numpy array of shape (N, W, H)
        """

        if isinstance(in_df_filename_or_list, pd.DataFrame):
            images = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2GRAY) for x in in_df_filename_or_list['fileloc']]
        elif isinstance(in_df_filename_or_list, list):
            images = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2GRAY) for x in in_df_filename_or_list]
            return images
        elif isinstance(in_df_filename_or_list, str):
            images = [cv2.cvtColor(cv2.imread(in_df_filename_or_list), cv2.COLOR_BGR2GRAY)]
        else:
            raise TypeError('in_df_filename_or_list can only be one of DataFrame, string or list')

        images = [cv2.resize(x, resize_shape, interpolation=cv2.INTER_CUBIC) for x in images]
        images = [x[chop[1]:-(chop[1] + 1), chop[0]:-(chop[0] + 1)] for x in images]
        images = np.stack(images, axis=0)

        # Convert from 0 to 1
        images = images/255.0

        return images

    # noinspection PyUnresolvedReferences
    @staticmethod
    def load_image(in_df_filename_or_list):
        """
        Reads the images into numpy arrays from the source

        :param in_df_filename_or_list: if DataFrame then reads the fileloc colum of the DataFrame and creates an
            image column. If string then assumes that is a filename(with path) and returns numpy array. If list then
            assumes a list of filenames(with path) and returns a list of numpy arrays
        :return:

        .. code-block:: python
            imgv = ImageLoader()
            imgv.load_n(10, defect_classes=None)
            dv = DefectViewer(imgv)
            dv.view_defects()
        """

        if isinstance(in_df_filename_or_list, pd.DataFrame):
            in_df_filename_or_list['image'] = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2GRAY)
                                               for x in in_df_filename_or_list['fileloc']]
            return in_df_filename_or_list
        elif isinstance(in_df_filename_or_list, list):
            images = [cv2.imread(x) for x in in_df_filename_or_list]
            return images
        elif isinstance(in_df_filename_or_list, str):
            return cv2.imread(in_df_filename_or_list)
        else:
            raise TypeError('in_df_filename_or_list can only be one of DataFrame, string or list')

    @staticmethod
    def get_coords(coord_list):
        # Returns the coordinates from the annotations_df 'bounding_box' column when using universal bounding boxes.

        # Assumes format is [xmin, xmax, ymin, ymax]
        xmin = coord_list[0]
        xmax = coord_list[1]
        ymin = coord_list[2]
        ymax = coord_list[3]

        return [xmin, ymin, xmax - xmin, ymax - ymin]

    @staticmethod
    def draw_rectangular_patch(x, y, w, h, color_='red', alpha_=0.2):
        # Creates a rectanglar bounding box
        return patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor=color_, facecolor='none', alpha=alpha_
        )

    @staticmethod
    def draw_polygon_patch(x, y, color_='green', alpha_=0.2):
        # Creates a polygon bounding box
        vector = np.column_stack((np.array(x), np.array(y)))
        poly = patches.Polygon(
            vector, color=color_, alpha=alpha_
        )
        return poly

    @staticmethod
    def draw_circle_patch(xy, radius, color_='blue', alpha_=0.2):
        # Creates a circular bounding region
        circle = patches.Circle(
            xy, radius, color=color_, alpha=alpha_
        )
        return circle

    @staticmethod
    def draw_elliptical_patch(xy, width, height, color_='yellow', alpha_=0.2, angle=0):
        # Creates a ellptical bounding region
        ellipse = patches.Ellipse(
            xy, width, height, angle, color=color_, alpha=alpha_
        )
        return ellipse

    @staticmethod
    def check_buffer(x, y, w, h, img, buffer):
        # Check that adding some buffer pixels does not go
        # outside the image bounds
        xmax = img.shape[1]
        ymax = img.shape[0]
        # Check x min
        if x - buffer < 0:
            x_ = 0
        else:
            x_ = x - buffer

        # Check y min
        if y - buffer < 0:
            y_ = 0
        else:
            y_ = y - buffer

        # Check x max
        if x + w + buffer > xmax:
            w_ = x + w
        else:
            w_ = x + w + buffer

        # Check y max
        if y + h + buffer > ymax:
            h_ = y + h
        else:
            h_ = y + h + buffer

        return x_, y_, w_, h_

    def _draw_annotation(self, ax, annotation_type, annotation):
        """

        :param ax:
        :param annotation_type:
        :param annotation:
        :return:
        """

        if annotation_type == 'bounding_box':
            # Create a rectangular patch
            rect = self.draw_rectangular_patch(*self.get_coords(eval(annotation)))
            ax.add_patch(rect)

        elif annotation_type == 'segmentations':
            # load the dictionary processed coordinates
            seg_dict = json.loads(annotation)
            if not seg_dict:
                return

            shape = seg_dict['name']

            if shape == 'polygon':
                all_x = seg_dict['all_points_x']
                all_y = seg_dict['all_points_y']
                polygon = self.draw_polygon_patch(all_x, all_y)
                ax.add_patch(polygon)

            elif shape == 'circle':
                xy = (seg_dict['cx'], seg_dict['cy'])
                radius = seg_dict['r']
                circle = self.draw_circle_patch(xy, radius)
                ax.add_patch(circle)

            elif shape == 'elipse':
                xy = (seg_dict['cx'], seg_dict['cy'])
                width = seg_dict['rx']
                height = seg_dict['ry']
                ellipse = self.draw_elliptical_patch(xy, width, height)
                ax.add_patch(ellipse)

            elif shape == 'rect':
                rect = self.draw_rectangular_patch(seg_dict['x'], seg_dict['y'], seg_dict['width'], seg_dict['height'])
                ax.add_patch(rect)

            elif shape == 'none':
                pass

    def view_defects(self, sample_df=None, group_by='defect_class', annotation_type='segmentations'):
        """
        View defect image without annotation and with annotations, side by side. Mark all annotations of the image

        :param sample_df:
        :param group_by:
        :param annotation_type:
        :return:
        """
        if self.il_obj is None:
            raise ValueError('Object of class Imageloader is a required input when view_defects function is used')

        # If samples are not provided then use the ones in ImadeLoader object
        sample_df = self.il_obj.sample_df if sample_df is None else sample_df
        if sample_df.empty:
            raise ValueError('Samples must either be provided as an input to this function or must be populated '
                             'in the ImageLoader class. Use method load_n in ImageLoader to create n image samples')

        if annotation_type == 'bounding_box':
            annotation_column = 'bounding_box_coords'
        elif annotation_type == 'segmentations':
            annotation_column = 'region_shape_attributes'
        else:
            raise KeyError('Annotation type can only be one of bounding_box or segmentations')

        # Load 'n' images
        sample_df = self.load_image(sample_df)

        # Create a separate visualization for each class
        for group_name, group_df in sample_df.groupby(group_by):

            #  Create multiple pairs of images
            fig = plt.figure(figsize=(6.4*2, 4.8*group_df.shape[0]))
            plt.suptitle(group_name)

            # Reset the index
            group_df = group_df.reset_index()

            for index, row in group_df.iterrows():

                # Get the filename for the image
                filename = row['filename']

                # Get all the annotations for this file
                this_file_df = self.il_obj.main_df[self.il_obj.main_df['filename'] == filename]

                # Keep only the annotations belonging to this class
                this_file_df = this_file_df[this_file_df['defect_class'] == group_name]

                # Add a subplot to the figure
                # noinspection PyTypeChecker
                ax = fig.add_subplot(group_df.shape[0], 2, index*2 + 1)

                # Show
                ax.imshow(row['image'], cmap='gray', norm=Normalize(vmin=0, vmax=1, clip=True))
                ax.set_title(filename)

                # Add a subplot to the figure
                # noinspection PyTypeChecker
                ax = fig.add_subplot(group_df.shape[0], 2, index*2 + 2)

                # Show the defect annotations
                ax.imshow(row['image'], cmap='gray', norm=Normalize(vmin=0, vmax=1, clip=True) )
                ax.set_title(group_name)

                for annotation in this_file_df[annotation_column]:
                    self._draw_annotation(ax, annotation_type, annotation)

        plt.show()


class Show:
    """

    """

    def __init__(self, save_filename=None, do_show=True, num_images=None, seed=None):
        """

        :param save_filename: Filename to save the output
        :param do_show: Display a plot or not
        :param num_images: If None then all images, otherwise the number listed
        :param seed: Used for shuffling indices when show is used with num_images
        """

        self.save_filename = save_filename
        self.do_show = do_show
        self.num_images = num_images
        self.seed = seed

    @staticmethod
    def _chk_shape(in_imgs):
        if len(in_imgs.shape) == 2:
            in_imgs = in_imgs[np.newaxis, :]

        return in_imgs

    def __lshift__(self, in_imw):
        """

        :param in_imw: ImageWrapper object or Iterable of ImageWrapper objects
        :return:
        """

        if not isinstance(in_imw, Iterable):
            in_imw = [in_imw, ]

        in_imgs = [x.images for x in in_imw]

        # Any one ImageWrapper object has the required image filenames
        self.show(in_imgs, category=[x.category for x in in_imw], image_labels=in_imw[0].image_labels)

        return in_imw

    def show(self, in_imgs, category=None, image_labels=None):
        """

        :param in_imgs: List or tuple of numpy array of shape (W, H) or (N, W, H)
        :param category: Image category(title)
        :param image_labels: Label for each sample of image
        :return:
        """

        if isinstance(in_imgs, np.ndarray):
            in_imgs = [in_imgs, ]

        # Number of cols and number of rows
        n_cols = len(in_imgs)
        self.num_images = in_imgs[0].shape[0] if self.num_images is None else self.num_images

        n_rows = min(in_imgs[0].shape[0], self.num_images)
        fig = plt.figure(figsize=(6.4*n_cols, 4.8*n_rows))

        # Which images to plot
        total_rows = in_imgs[0].shape[0]
        accepted_rows = list(range(total_rows))
        if self.seed is not None:
            random.seed(self.seed)
        shuf(accepted_rows)

        if self.num_images is not None:
            accepted_rows = accepted_rows[:self.num_images]

        # Assumes every item on the list has the same number of dimensions
        # Walk through every image
        plt_rows = 0
        for row_cnt in range(total_rows):
            # Only those rows that were accepted are plotted
            if row_cnt not in accepted_rows:
                continue

            # Walk through every column of the image
            for col_cnt in range(len(in_imgs)):
                img_cnt = plt_rows*n_cols + col_cnt + 1
                ax = fig.add_subplot(n_rows, n_cols, img_cnt)
                ax.imshow(np.squeeze(in_imgs[col_cnt][row_cnt, :, :]), cmap='gray',
                          norm=Normalize(vmin=0, vmax=1, clip=True))

                if col_cnt == 0 and image_labels is not None:
                    ax.set_ylabel(chunk(image_labels[row_cnt]), size='large')

                # Add a column label if row is 0
                if plt_rows == 0 and category is not None:
                    ax.set_title(category[col_cnt])

            plt_rows += 1

        plt.tight_layout()
        if self.save_filename is not None:
            plt.savefig(self.save_filename)

        if self.do_show:
            plt.show()
        else:
            plt.close()

        return in_imgs


class Exposure:

    def __init__(self, mode='histo', consume_kwargs=True, **kwargs):
        """

        :param mode: Histo, Gamma,
        :param consume_kwargs:
        :param kwargs:
        """

        accepted_modes = ['stretch', 'histo', 'adaptive', 'sigmoid', 'gamma', 'invert', 'mean_norm', 'dynamic_sigmoid']
        if mode not in accepted_modes:
            raise KeyError(f'Unsupported mode, it should be one of {accepted_modes}')
        self.mode = mode

        self.params = {}
        # indict, key, default, out_dict, exception=False
        if mode == "adaptive":
            # Check if kernel_size input is provided
            input_check(kwargs, 'kernel_size', None, self.params, exception=False)

            # Check if clip input is provided
            input_check(kwargs, 'clip_limit', None, self.params, exception=False)

            # Check if clip input is provided
            input_check(kwargs, 'nbins', None, self.params, exception=False)
        elif mode == 'sigmoid' or mode == 'dynamic_sigmoid':
            # Check if kernel_size input is provided
            input_check(kwargs, 'cutoff', 0.5, self.params, exception=False)

            # Check if clip input is provided
            input_check(kwargs, 'gain', 10, self.params, exception=False)

            # Check if clip input is provided
            input_check(kwargs, 'inverse', False, self.params, exception=False)
        elif mode == 'gamma':
            # Check if kernel_size input is provided
            input_check(kwargs, 'gamma', 1, self.params, exception=True)

            # Check if clip input is provided
            input_check(kwargs, 'gain', 1, self.params, exception=False)

        if consume_kwargs and kwargs:
            raise KeyError(f'Unused keys in kwargs {kwargs.keys()}')

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

        # This is the processed nd array
        out_imgs = self.apply(in_imgs)

        # Add the
        category = f'\n Exposure mode {self.mode} with params {self.params}' if self.params else \
                   f'\n Exposure mode {self.mode}'
        category = in_imw.category + line_split_string(category)

        out_imw = ImageWrapper(out_imgs, category=category, image_labels=copy.deepcopy(in_imw.image_labels))

        return in_imw, out_imw

    def apply(self, in_imgs):

        if self.mode == 'histo':
            return self.histogram_equalization(in_imgs)
        elif self.mode == 'stretch':
            return self.contrast_stretching(in_imgs)
        elif self.mode == 'sigmoid':
            return self.adjust_sigmoid(in_imgs)
        elif self.mode == 'dynamic_sigmoid':
            return self.adjust_sigmoid_dynamic(in_imgs)
        elif self.mode == 'gamma':
            return self.adjust_gamma(in_imgs)
        elif self.mode == 'adaptive':
            return self.adaptive_histogram_equalization(in_imgs)
        elif self.mode == 'invert':
            return self.invert(in_imgs)
        elif self.mode == 'mean_norm':
            return self.mean_norm(in_imgs)
        else:
            raise KeyError('Hmm... something went wrong')

    @staticmethod
    def contrast_stretching(in_imgs):
        """
        Stretches the values of the pixels from 0 to 1, irrespective of teh starting scale

        :param in_imgs: Input images of shape (N, W, H)
        :return:
        """

        all_min = np.min(in_imgs, axis=(-2, -1), keepdims=True)
        all_max = np.max(in_imgs, axis=(-2, -1), keepdims=True)
        out_img = (in_imgs - all_min) / (all_max - all_min)

        return out_img

    def adjust_sigmoid(self, in_imgs):
        """
        Performs a sigmoid correction on the input image

        :param in_imgs: Input images of shape (N, W, H)
        :return:
        """

        # Sign for the sigmoid
        sign = -1 if self.params['inverse'] else 1

        # Cutoff point for the sigmoid
        cutoff = self.params['cutoff']

        # Gain parameter controls the steepness of the sigmoid
        gain = self.params['gain']

        return 1/(1 + np.exp(-sign * gain * (in_imgs - cutoff)))
    
    def adjust_sigmoid_dynamic(self, in_imgs):
        """
        Performs a sigmoid correction on the input image

        :param in_imgs: Input images of shape (N, W, H)
        :return:
        """

        # Sign for the sigmoid
        sign = -1 if self.params['inverse'] else 1

        # Cutoff point for the sigmoid
        cutoff = self.params['cutoff']

        # Gain parameter controls the steepness of the sigmoid
        gain = self.params['gain']
        max_vals = in_imgs.max(axis=(-2, -1))
        min_vals = in_imgs.min(axis=(-2, -1))
        out_imgs = []
        for img, maxv, minv in zip(in_imgs, max_vals, min_vals):
            # Data range
            val_range = maxv - minv
            # Fraction at which to make the cutoff
            cut_val = minv + (val_range * cutoff)
            # This is the sigmoid adjusted image
            adjusted = 1/(1 + np.exp(-sign * gain * (img - cut_val)))
            out_imgs.append(adjusted)
        out_imgs = np.stack(out_imgs)

        return out_imgs

    def adjust_gamma(self, in_imgs):
        """
        Performs a sigmoid correction on the input image

        :param in_imgs: Input images of shape (N, W, H)
        :return:
        """

        # Gamma value
        gamma = self.params['gamma']

        # Gain parameter controls the steepness of the gamma
        gain = self.params['gain']

        return gain*(in_imgs**gamma)

    @staticmethod
    def invert(in_imgs):
        """
        Inverts the brightness of the image

        :param in_imgs: Input images of shape (N, W, H)
        :return:
        """

        in_imgs = 1 - in_imgs

        return in_imgs

    @staticmethod
    def mean_norm(in_imgs):
        """
        Shifts the image such that the mean is at 0.5

        :param in_imgs: Input images of shape (N, W, H)
        :return:
        """

        mean = np.mean(in_imgs, axis=(-2, -1), keepdims=True)
        in_imgs += (0.5 - mean)

        # Ensure the results are within bound
        in_imgs[in_imgs < 0] = 0
        in_imgs[in_imgs > 1] = 1

        return in_imgs

    def adaptive_histogram_equalization(self, in_imgs):
        """
        Contrast limited adaptive histogram equalization of each image

        :param in_imgs: Input images of shape (N, W, H)
        :return:
        """

        # Convert the input to integer datatype
        in_imgs = (in_imgs * 255).astype('uint8')

        # Returns the counts per axis
        equalized_histo = [exposure.equalize_adapthist(x, **self.params) for x in in_imgs]

        return np.stack(equalized_histo, axis=0)

    def histogram_equalization(self, in_imgs):
        """
        Standard histogram equalization of each image

        :param in_imgs: Input images of shape (N, W, H)
        :return:
        """

        # Convert the input to integer datatype
        in_imgs = (in_imgs * 255).astype('uint8')

        # Returns the counts per axis
        equalized_histo = [exposure.equalize_hist(x, **self.params) for x in in_imgs]

        return np.stack(equalized_histo, axis=0)


if __name__ == '__main__':
    imgv = ImageLoader()

    do_test_equalization = True
    if do_test_equalization:
        sdf = imgv.load_n(1, defect_classes='FrontGridInterruption')
        dv = DefectViewer(imgv)

        imgs = DefectViewer() << (ImageLoader(defect_class='FrontGridInterruption') << 50)
        Show(num_images=3, seed=43) << imgs

        # Equalize using sigmoid
        # eq = Equalize(mode='sigmoid', gain=10, cutoff=0.5, inverse=True)
        # eq = Equalize(mode='gamma', gain=1, gamma=0.2)
        Show(num_images=3, seed=43) << (Exposure(mode='dynamic_sigmoid') << imgs)

        # eq.histogram_equalization(imgs)
        # eq.adaptive_histogram_equalization(imgs)

    do_test_loader = False
    if do_test_loader:
        # Load all defects
        all_defects = imgv.load_n(10, defect_classes=None)

        # Load one defect, input is a string
        one_defect = imgv.load_n(10, defect_classes=imgv.defect_classes[0])

        # Load one defect, input is a string
        two_defects = imgv.load_n(10, defect_classes=imgv.defect_classes[:2])

        if all_defects.empty or one_defect.empty or two_defects.empty:
            raise ValueError('Unexpectedly, one or more of the DataFrames was empty')

    do_test_viewer = False
    if do_test_viewer:

        # Load all defects
        imgv.load_n(10, defect_classes=None)
        dv = DefectViewer(imgv)
        dv.view_defects()

    print('Complete')
