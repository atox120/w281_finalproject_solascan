import os
import time
import shutil
import numpy as np
import pandas as pd
from random import shuffle, seed

# Missing files source
missing_source = "/home/aswin/Documents/MIDS/w281/UCF-EL-Defect/Test_Images"
train_images_path = '/home/aswin/Documents/MIDS/w281/w281_finalproject_solascan/data/images/train'
test_images_path = '/home/aswin/Documents/MIDS/w281/w281_finalproject_solascan/data/images/test'


def prepare_df():
    # Read into DataFrames
    annotations_df = pd.read_csv('processed_annotations.csv')
    train_df = pd.read_csv('train_images.csv')
    train_df = train_df.rename(columns={'imagename': 'filename'})
    train_df = train_df[['filename']]

    test_df = pd.read_csv('test_images.csv')
    test_df = test_df.rename(columns={'imagename': 'filename'})
    test_df = test_df[['filename']]

    # Find the missing files
    both_df = pd.concat((train_df, test_df))

    # The missing fellas
    missing_df = annotations_df[np.logical_not(annotations_df['filename'].isin(both_df['filename']))]
    missing_df = missing_df[missing_df['defect_class'] == 'None']

    # Now move these files randomly between train
    missing_df = missing_df.reset_index()

    # Get only one row per file
    missing_df = missing_df.groupby('filename').head(1)
    index = list(range(missing_df.shape[0]))
    seed(2**21)
    shuffle(index)

    train_stop = int(len(index) * 0.8)
    annotations_df = annotations_df.set_index('filename')

    train_missing = missing_df.iloc[index[:train_stop]]
    test_missing = missing_df.iloc[index[train_stop:]]

    train_df = pd.concat((train_df, train_missing[train_df.columns]))
    test_df = pd.concat((test_df, test_missing[test_df.columns]))

    train_df.to_csv('train.csv')
    test_df.to_csv('test.csv')

    return annotations_df, train_df, test_df


def move_files(src_path, dest_path, df, annotations_df):
    # Move the file from source to train destination
    total_count = 0
    for filename in df['filename']:
        dest_file = os.path.join(dest_path, filename)
        source_file = os.path.join(src_path, filename)

        # Run the
        if not os.path.exists(source_file):
            print(f'## {filename} - file not found at source')
        else:
            if not os.path.exists(dest_file):
                defect_class = annotations_df.loc[filename]['defect_class']
                if not isinstance(defect_class, str):
                    defect_class = defect_class.unique().tolist()
                    defect_class = [x for x in defect_class if x != 'None']

                    if len(defect_class) == 0:
                        defect_class = 'None'
                    else:
                        defect_class = str(defect_class)

                if defect_class != 'None':
                    print(f"{filename} not at dest is of class {defect_class}")

                #
                shutil.copy(source_file, dest_file)

                total_count += 1


def check_for_incorrect_files(df, file_path):

    # Check if there are any files in the train folder that don't belong there
    all_files = os.listdir(file_path)
    file_list = df['filename'].to_list()

    total_count = 0
    for filename in all_files:
        if filename not in file_list:
            print(filename)
            total_count += 1
            os.remove(os.path.join(file_path, filename))
            time.sleep(0.01)

    print(total_count)


if __name__ == '__main__':
    ann_df, tr_df, tst_df = prepare_df()

    print('Moving files for Train')
    move_files(missing_source, train_images_path, tr_df, ann_df)

    print('Moving files for Test')
    move_files(missing_source, test_images_path, tst_df, ann_df)

    print('Checking for incorrect files Train')
    check_for_incorrect_files(tr_df, train_images_path)

    print('Checking for incorrect files Test')
    check_for_incorrect_files(tst_df, test_images_path)