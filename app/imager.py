##### Imports Packages #####

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import random
import json

##### CLASSES #####

## Image Loader ##
class image_loader():
    
    def __init__(self):
     
        #Where the processed annotations csv file is stored. 
        self.processed_annotation_path = "../data/processed_annotations.csv"
        # Main df
        self.annotations_df = pd.read_csv(self.processed_annotation_path)
        # List of all defect classes
        self.defect_classes = self.annotations_df['defect_class'].unique().tolist()
        # List of all filenames
        self.all_files = self.annotations_df['filename'].unique().tolist()
        # Count of each class in complete dataset.
        self.instance_count = dict(self.annotations_df['defect_class'].value_counts())
        
        # Instatiate some attributes - i think this is not neccesary...
        self.example_dict = {}
        self.file_list = []
        self.bounding_boxes = []
        self.annotation_shape = []
        self.segmentations = []
        self.vector_list = []
        self.file_list_multiple_annotations = []
        
    def n_instances(self, defect_class):
    # returns the number of instances in a given defect class
        
        return self.instance_count[defect_class]
    
    def load_n_images(self, n, shuffle=True, retn = True):
        
        ## NOT WORKING YET
        ## Loads n images
        ## Returns a dict with schema:
        ## 
        ##  "filename":[bounding_boxes, annotation_shape, segmentations]"
        ## 
        ## filename -> source image file. 
        ## bounding_boxes -> extracted bounding box from the original segmentations
        ## annotation_shape -> the original segmentation shape (rect, circle, elipse, polygon).
        ## segmentations -> the original segmentation coordinates. 
        ## This schema is repeated for n filenames. 
        
        #Clear any old data
        self.example_dict = {}
              
        # Check that there are >=n examples within that class:
        if n >= len(self.all_files):
            raise Exception(f"There are only {len(self.all_files)} available files. Pick a smaller n'")
        #Create an index:
        n_elements = len(self.all_files)
        idx = range(0, n_elements)
        
        #Shuffle if required
        if shuffle:
            idx = [x for x in random.sample(idx, k = n_elements)][:n]
        else:
            idx = idx[:n]
            
        # Get a list of unique filenames for filtering:
        self.file_list = np.array(self.all_files)[idx]
        
        self.file_list_multiple_annotations = self.annotations_df['filename']\
            [self.annotations_df['filename'].isin(self.file_list)]\
            .tolist()
        
        # Get original segmentations
        self.segmentations = self.annotations_df['region_shape_attributes']\
            [self.annotations_df['filename'].isin(self.file_list)]\
            .tolist()
        
        # Get a list of bounding boxes
        self.bounding_boxes = self.annotations_df['bounding_box_coords']\
            [self.annotations_df['filename'].isin(self.file_list)]\
            .tolist()
        
        #I don't know why, but it is loading this list of lists as a list of strings.
        # This converts each string to a list. 
        self.bounding_boxes = [json.loads(x) for x in self.bounding_boxes]

        # Get a list of shapes
        self.annotation_shape = self.annotations_df['annotation_shape']\
            [self.annotations_df['filename'].isin(self.file_list)]\
            .tolist()
        
        # Write each example to the dictionary with attribute
        ### THIS FAILS - can't have multiple keys with the same filename!!!! #####
        for i in range(len(self.file_list_multiple_annotations)):
            self.example_dict.update(
                {f"{self.file_list_multiple_annotations[i]}":[
                    self.bounding_boxes[i], 
                    self.annotation_shape[i],
                    self.segmentations[i],
                ]}
            )
        
        if retn:
            return self.example_dict
        
    def load_n_examples(self, n, defect_class, shuffle=True, retn = True):
        
        ## Loads n examples of a given defect class from 
        ## either consecutive or shuffled instances from the dataset. 
        ## Returns a dict with schema:
        ## 
        ##  "filename":[bounding_boxes, annotation_shape, segmentations]"
        ## 
        ## filename -> source image file. 
        ## bounding_boxes -> extracted bounding box from the original segmentations
        ## annotation_shape -> the original segmentation shape (rect, circle, elipse, polygon).
        ## segmentations -> the original segmentation coordinates. 
        ## This schema is repeated for n filenames. 
        
        #Clear any old data
        self.example_dict = {}
        
        # Check that the defect_class is in the df:
        if defect_class not in self.defect_classes:
            raise Exception(f"Class titled {defect_class} not found.")
        
        # Check that there are >=n examples within that class:
        if n > self.n_instances(defect_class):
            raise Exception(f"There are only {self.n_instance(defect_class)} examples of {defect_class}'s. Pick a smaller n'")
            
        #Create an index:
        n_elements = self.n_instances(defect_class)
        idx = range(0, n_elements)
        
        #Shuffle if required
        if shuffle:
            idx = [x for x in random.sample(idx, k = n_elements)][:n]
        else:
            idx = idx[:n]
            
        # Get a list of filenames       
        self.file_list = self.annotations_df['filename']\
            [self.annotations_df['defect_class']==defect_class]\
            .iloc[idx]\
            .tolist()
        
        # Get original segmentations
        self.segmentations = self.annotations_df['region_shape_attributes']\
            [self.annotations_df['defect_class']==defect_class]\
            .iloc[idx]\
            .tolist()
        
        # Get a list of bounding boxes
        self.bounding_boxes = self.annotations_df['bounding_box_coords']\
        [self.annotations_df['defect_class']==defect_class]\
            .iloc[idx]\
            .tolist()
        
        #I don't know why, but it is loading this list of lists as a list of strings.
        # This converts each string to a list. 
        self.bounding_boxes = [json.loads(x) for x in self.bounding_boxes]

        # Get a list of shapes
        self.annotation_shape = self.annotations_df['annotation_shape']\
            [self.annotations_df['defect_class']==defect_class]\
            .iloc[idx]\
            .tolist()
        
        # Write each example to the dictionary with attribute
        for i in range(len(self.file_list)):
            self.example_dict.update(
                {f"{self.file_list[i]}":[
                    self.bounding_boxes[i], 
                    self.annotation_shape[i],
                    self.segmentations[i],
                ]}
            )
        
        if retn:
            return self.example_dict
        
    def load_image_vectors(self, pth=f"../data/images/", retn = True):
        
        ## Loads each file in the file_list as a numpy array, and returns the list of 
        # images as a list of vectors.
                
        if self.file_list == []: 
            print("file_list is empty, please pass an example dict or run image_loader.load_n_examples(n_examples, defect_class)")
        else:
            self.vector_list = [load_image(pth, img_name) for img_name in self.file_list]

        if retn:
            return self.vector_list

        
## Defect_Viewer ##        
class defect_viewer():
    
    def __init__(self, image_loader=None):
        
        if image_loader == None:
            print('Need to pass a loaded image_loader')
        else:
            self.image_loader = image_loader
        
            #Load image information
            self.file_list = self.load_file_list()
            self.example_dict = self.load_example_dict()
            self.vector_list = self.load_vector_list()        
        
    def load_file_list(self):
        
        if self.image_loader.file_list == []:
            print('Error - No defects loaded into the image loader. Run self.image_loader.load_n_examples()')
            return []
        else:
            return self.image_loader.file_list
    
    def load_vector_list(self):
        
        if self.image_loader.vector_list == []:
            print('Error - No images loaded into the image array. Run self.image_loader.load_image_vectors()')
            return []
        else:
            return self.image_loader.vector_list

    def load_example_dict(self):
        
        if self.image_loader.example_dict == {}:
            print('Error - No defects loaded into the image loader. Run self.load_n_examples()')
            return {}
        else:
            return self.image_loader.example_dict
        
    
    def view_loaded_files(self, annotation_type=None):
        
        if annotation_type == None:
            print("""Error: Please specify the type of Annotation highlights as \'bounding box\' or \'segmentations""")
        
        n_examples = len(self.image_loader.file_list)
        
        for i in range(n_examples):
            
            img = self.vector_list[i]
            filename = self.file_list[i]
                 
            # Load the bounding boxes using the get_coords function, which are 
            # stored in the first list element of the example_dict.
            x,w,y,h = get_coords(self.example_dict[f'{filename}'][0])

                #Plot
            fig, axs = plt.subplots(1,2,figsize=(10,10))
            axs[0].set_title(f"Full Image - {filename}")
            axs[0].imshow(img)
            
            if annotation_type == 'bounding_box':
                
                # Create a rectangular patch
                rect = draw_rectangular_patch(x,y,w,h)
                axs[0].add_patch(rect)

            elif annotation_type == 'segmentations':
                
                # load the dictionary processed coordinates
                seg_dict = json.loads(self.example_dict[f'{filename}'][2])
                shape = seg_dict['name']
                
                if shape == 'polygon':
                    all_x = seg_dict['all_points_x']
                    all_y = seg_dict['all_points_y']
                    polygon =  draw_polygon_patch(all_x, all_y)
                    axs[0].add_patch(polygon)
                
                elif shape == 'circle':
                    xy = (seg_dict['cx'],seg_dict['cy'])
                    radius = seg_dict['r']
                    circle = draw_circle_patch(xy, radius)
                    axs[0].add_patch(circle)
                    
                elif shape == 'elipse':
                    xy = (seg_dict['cx'],seg_dict['cy'])
                    width = seg_dict['rx']
                    height = seg_dict['ry']
                    ellipse = draw_elliptical_patch(xy, width, height, 0)
                    axs[0].add_patch(ellipse)                    
                
                elif shape == 'rect':
                    rect = draw_rectangular_patch(x,y,w,h)
                    axs[0].add_patch(rect)
                
                elif shape == 'none':
                    pass 
                
            
            #Plot cropped - note we use the coords from the 
            # bounding box calculations
            #buffer = 2
            #x,y,w,h = check_buffer(x,y,w,h,img, buffer)
                
            
            axs[1].imshow(img[y:y+h, x:x+w])
            axs[1].set_title('Cropped Image')
           
        
##### FUNCTIONS #####        

def load_image(img_pth, img_name):
    #Load an image given the path and filename
    #Concatenate path and filename
    pth = img_pth + img_name
    #read image
    img = plt.imread(pth)
    
    return img       

def get_coords(coord_list):
    #Returns the coordinates from the annotations_df 'bounding_box' column when using universal bounding boxes. 
    
    # Assumes format is [xmin, xmax, ymin, ymax]
    xmin = coord_list[0]
    xmax = coord_list[1]
    ymin = coord_list[2]
    ymax = coord_list[3]
    
    return [xmin, xmax-xmin, ymin, ymax-ymin]

def draw_rectangular_patch(x,y,w,h,color_='red', alpha_=0.2):
    # Creates a rectanglar bounding box    
    return patches.Rectangle(
        (x, y), w, h, linewidth=1, edgecolor=color_, facecolor='none', alpha=alpha_
)

def draw_polygon_patch(x, y, color_='green', alpha_=0.2):
    # Creates a polygon bounding box
    vector = np.column_stack((np.array(x), np.array(y)))
    poly = patches.Polygon(
        vector, color=color_, alpha=alpha_
    )
    return poly

def draw_circle_patch(xy, radius, color_='blue', alpha_=0.2):
    # Creates a circular bounding region
    circle = patches.Circle(
        xy, radius, color=color_, alpha=alpha_
    )    
    return circle

def draw_elliptical_patch(xy, width, height, color_='yellow', alpha_=0.2, angle=0):
    # Creates a ellptical bounding region
    ellipse = patches.Ellipse(
        xy, width, height, angle, color=color_, alpha=alpha_
    )    
    return ellipse

def check_buffer(x,y,w,h,img, buffer):
    
    # Check that adding some buffer pixels does not go 
    # outside the image bounds 
    
    xmin,ymin  = (0,0)
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


if __name__ == '__main__':

    image_loader()
    defect_viewer()
    