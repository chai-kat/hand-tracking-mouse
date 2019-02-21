import cv2
import tensorflow as tf
import numpy
import os
from scipy.io import loadmat

#! label of the form (xmin, ymin, xmax, ymax) - should be normalized
#! prediction of the form (xmin, ymin, xmax, ymax)

for folder in eval_folders:
    image = cv2.imread(image_path)
    height = image.shape[0] # Image height
    width = image.shape[1] # Image width
    channels = image.shape[2]
    
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)

    image_annotations = folder_mat[image_index]
    for hand in image_annotations:
    #check if there's actually points in the hands set
    if len(hand) > 1: 
        xmax, ymax = get_maxes(hand)
        xmin, ymin = get_mins(hand)
        class_name = "hand"
        class_id = 1

        #normalize all the points - i.e. x divided by width, y divided by height
        #no need to worry about integer truncation - the dataset is all floats anyway
        xmax = xmax / width
        xmin = xmin / width
        ymax = ymax / height
        ymin = ymin / height
        #append all the data to the above defined lists
        xmaxs.append(xmax)
        xmins.append(xmin)
        ymaxs.append(ymax)
        ymins.append(ymin)