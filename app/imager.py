import os
import cv2
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ImageLoader:
    """
    Easy loading, splitting into test and CV, and viewing of images


    """

    def __init__(self):

        # Where the processed annotations csv file is stored.
        this_file_location = os.path.abspath("")

        # This is the
        self.processed_annotation_path = os.path.join(this_file_location, "../data/processed_annotations.csv")
        self.train_files = os.path.join(this_file_location, "../data/labels/train_images.csv")
        self.test_files = os.path.join(this_file_location, "../data/labels/test_images.csv")
        self.train_file_loc = os.path.join(this_file_location, "../data/images/train")
        self.test_file_loc = os.path.join(this_file_location, "../data/images/test")

        # These are all the files in the train folder
        all_files = os.listdir(self.train_file_loc)

        # Load the multiple DataFrames
        self.annotations_df = pd.read_csv(self.processed_annotation_path)
        self.train_files_df = pd.read_csv(self.train_files)
        self.train_files_df = self.train_files_df.rename(columns={'imagename': 'filename',
                                                                  'annotatedname': 'annotated_filename'})

        # load
        defect_files = self.train_files_df['filename'].tolist()
        good_files = [x for x in all_files if x not in defect_files]
        # Create a DataFrame with no issues
        good_df = pd.DataFrame({'filename': good_files, 'annotated_filename': good_files})
        self.train_files_df = pd.concat((self.train_files_df, good_df))

        # Keep only one copy of the file and folder
        self.cv_files_df = pd.DataFrame()
        self.sample_df = pd.DataFrame()

        # Keep only rows with test data
        self.main_df = self.annotations_df.merge(self.train_files_df, on='filename', how='inner')
        self.main_df['sno'] = list(range(0, self.main_df.shape[0]))

        # List of all defect classes
        self.defect_classes = self.annotations_df['defect_class'].unique().tolist()

        # List of all filenames
        self.all_files = self.annotations_df['filename'].unique().tolist()

        # Count of each class in complete dataset.
        self.instance_count = dict(self.annotations_df['defect_class'].value_counts())

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

    def load_n(self, n, shuffle=True, defect_classes=None):
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
                                     'list of strong of defect classes or None(all classes)')
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
        self.sample_df = self.main_df.groupby('defect_class').head(n)
        self.sample_df = self.sample_df[['filename', 'bounding_box_coords',
                                         'annotation_shape', 'region_shape_attributes',
                                         'defect_class']]

        # If defect classes were provided then only keep the required ones
        self.sample_df = self.sample_df[self.sample_df['defect_class'].isin(defect_classes)]

        # Add the location of the files to read from
        self.sample_df['fileloc'] = [os.path.join(self.train_file_loc, x) for x in self.sample_df['filename']]

        return self.sample_df


class DefectViewer:
    """
    Visualizes defects and provides annotations in the form of bounding boxes or segmentations
    """

    def __init__(self, il_obj):
        """

        :param il_obj: Object of ImageLoader
        """

        self.il_obj = il_obj

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
                ax.imshow(row['image'], cmap='gray')
                ax.set_title(filename)

                # Add a subplot to the figure
                # noinspection PyTypeChecker
                ax = fig.add_subplot(group_df.shape[0], 2, index*2 + 2)

                # Show the defect annotations
                ax.imshow(row['image'], cmap='gray')
                ax.set_title(group_name)

                for annotation in this_file_df[annotation_column]:
                    self._draw_annotation(ax, annotation_type, annotation)

        plt.show()


if __name__ == '__main__':
    imgv = ImageLoader()

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

    do_test_viewer = True
    if do_test_viewer:

        # Load all defects
        imgv.load_n(10, defect_classes=None)
        dv = DefectViewer(imgv)
        dv.view_defects()

    print('Complete')
