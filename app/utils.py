## Utility functions

#Imports
import math

class ImageWrapper:

    def __init__(self, images, image_labels=None, category=None):

        self.images = images
        self.category = category
        self.image_labels = image_labels


def input_check(indict, key, default, out_dict, exception=False):
    """
    Checks that the required inputs are parsed to the output dictionary.
    If exception = True, then this is a required input and an error is raised. 
    Else default parameter is assigned to that parameter.
    """

    try:
        out_dict[key] = indict[key]
        del indict[key]
    except KeyError:
        if exception:
            raise KeyError(f'{key} is a required input and was not provided')
        else:
            if default is not None:
                out_dict[key] = default


def line_split_string(instr, delimiter=' '):
    instr = str(instr)
    # instr = instr.replace('/n', ' ')

    split_strng = instr.split(delimiter)
    out_strng = ' '

    new_length = 0
    for strng in split_strng:
        if new_length > 40:
            out_strng += '\n'
            new_length = 0
        new_length += len(strng)

        out_strng += f' {strng}'
    return out_strng


def chunk(instr):

    instr = str(instr)
    num_chunks = math.ceil(len(instr)/40)
    split_strng = [instr[i:i+40] for i in range(num_chunks)]

    out_strng = ''
    for strng in split_strng:
        out_strng += strng + '\n'

    return out_strng[:-2]
