#want to get filename,width,height,class,xmin,ymin,xmax,y,max
#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
#Adapted from the above code
#https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py#L26
# Also credit to Dat Tran for the code on getting the encoded image bytes

import tensorflow as tf
import numpy as np
import cv2
import os
import io
from object_detection.utils import dataset_util
from scipy.io import loadmat


WHERE_IS_SAMPLES_DIR = ""
SAMPLES_DIR = "{}/egohands_data/_LABELLED_SAMPLES".format(WHERE_IS_SAMPLES_DIR)
cwd = os.getcwd()
os.mkdir(os.join(cwd, "DatasetTFRecords"))
#organize the sub-directories and their paths into a list
SUB_DIRECTORIES = ['CARDS_COURTYARD_B_T', 'CARDS_COURTYARD_H_S', 'CARDS_COURTYARD_S_H', 'CARDS_COURTYARD_T_B', 'CARDS_LIVINGROOM_B_T', 'CARDS_LIVINGROOM_H_S', 'CARDS_LIVINGROOM_S_H', 'CARDS_LIVINGROOM_T_B', 'CARDS_OFFICE_B_S', 'CARDS_OFFICE_H_T', 'CARDS_OFFICE_S_B', 'CARDS_OFFICE_T_H', 'CHESS_COURTYARD_B_T', 'CHESS_COURTYARD_H_S', 'CHESS_COURTYARD_S_H', 'CHESS_COURTYARD_T_B', 'CHESS_LIVINGROOM_B_S', 'CHESS_LIVINGROOM_H_T', 'CHESS_LIVINGROOM_S_B', 'CHESS_LIVINGROOM_T_H', 'CHESS_OFFICE_B_S', 'CHESS_OFFICE_H_T', 'CHESS_OFFICE_S_B', 'CHESS_OFFICE_T_H', 'JENGA_COURTYARD_B_H', 'JENGA_COURTYARD_H_B', 'JENGA_COURTYARD_S_T', 'JENGA_COURTYARD_T_S', 'JENGA_LIVINGROOM_B_H', 'JENGA_LIVINGROOM_H_B', 'JENGA_LIVINGROOM_S_T', 'JENGA_LIVINGROOM_T_S', 'JENGA_OFFICE_B_S', 'JENGA_OFFICE_H_T', 'JENGA_OFFICE_S_B', 'JENGA_OFFICE_T_H', 'PUZZLE_COURTYARD_B_S', 'PUZZLE_COURTYARD_H_T', 'PUZZLE_COURTYARD_S_B', 'PUZZLE_COURTYARD_T_H', 'PUZZLE_LIVINGROOM_B_T', 'PUZZLE_LIVINGROOM_H_S', 'PUZZLE_LIVINGROOM_S_H', 'PUZZLE_LIVINGROOM_T_B', 'PUZZLE_OFFICE_B_H', 'PUZZLE_OFFICE_H_B', 'PUZZLE_OFFICE_S_T', 'PUZZLE_OFFICE_T_S']
for index, folder_name in enumerate(SUB_DIRECTORIES):
  SUB_DIRECTORIES[index] = SAMPLES_DIR + "/" + folder_name

#organize the image paths into folder
sample_files = []
for folder_path in SUB_DIRECTORIES:
    dir_contents = os.listdir(folder_path)
    #leave only jpg names
    for item in dir_contents:
        if not item.endswith("jpg"):
            dir_contents.remove(item)
    #sort contents the way you'd see in a file manager
    dir_contents.sort()
    #make each item a path instead of just filename
    for x in range(0, len(dir_contents)):
        dir_contents[x] = folder_path + "/" + dir_contents[x]
    sample_files.append(dir_contents)


#split the sample files dir up
train_sample_files = sample_files[:-6]
validation_sample_files = sample_files[-6:-2]
eval_sample_files = sample_files[-2:]

def clean_polygon_mat(polygon_path):
  mat_dict = loadmat(polygon_path)
  mat = mat_dict["polygons"]
  mat = mat[0]
  mat = np.asarray(mat)
  return mat

def get_maxes(points):
    #let points be only one hand, call this four times (one for each hand array)
    #make sure len(points) is greater than 0, sometimes the arrays are empty (no such hand)
    #gets index of xmax,ymax respectively
    maxes = np.argmax(points, axis=0)
    #return in the form of xmax,ymax
    xmax = points[maxes[0], 0]
    ymax = points[maxes[1], 1]
    return xmax, ymax

def get_mins(points):
    #let points be only one hand, call this four times (one for each hand array)
    #make sure len(points) is greater than 0, sometimes the arrays are empty (no such hand)
    #gets index of xmin,ymin respectively
    mins = np.argmin(points, axis=0)
    #return in the form of xmin,ymin
    xmin = points[mins[0], 0]
    ymin = points[mins[1], 1]
    return xmin, ymin

def create_tf_example(folder_mat, image_path, image_index):
  image = cv2.imread(image_path)
  height = image.shape[0] # Image height
  width = image.shape[1] # Image width
  channels = image.shape[2]

  filename = image_path # Filename of the image. Empty if image is not from file
  #code from Dat Tran for opening the encoded image
  with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read() # Encoded image bytes

  image_format = b'jpeg'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

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
      classes_text.append(class_name)
      classes.append(class_id)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def main(_):
  #run three separate batches for train, test/validation, and eval
  #we only need train and test for training the model, 
  #but we'll use eval to find the final accuracy of the trained model

  #for train
  print "working on the train split"
  last_folder = 0
  write_path = cwd + "DatasetTFRecords/train.record"
  writer = tf.python_io.TFRecordWriter(write_path)
  for folder_index, folder in enumerate(train_sample_files):
    print "Working on " + SUB_DIRECTORIES[folder_index]
    print "Part " + str((folder_index+1)) + " of 48"
    #since we've taken folders 0-41 we don't need to add anything to the folder_index
    polygon_path = SUB_DIRECTORIES[folder_index] + "/polygons.mat"
    polygon_file = clean_polygon_mat(polygon_path)
    for image_index, image_path in enumerate(folder):
      tf_example = create_tf_example(polygon_file, image_path , image_index)
      writer.write(tf_example.SerializeToString())
    last_folder += 1
  writer.close()

  #for test/validation
  print "working on the test split"
  last_folder1 = last_folder #we can't change the value of last_folder while the next loop is running 
                             #so we need to make a new variable to store it
  write_path = cwd + "/DatasetTFRecords/test.record"
  writer = tf.python_io.TFRecordWriter(write_path)
  for folder_index, folder in enumerate(validation_sample_files):
    #since we've taken folders 42-45 we need to add 42 to the folder_index to get us to the right place
    #however a better solution is to count the number of folders we've been through in the last loop 
    #this means we can change the splits without having to come back down here and worry about the increments (lines 45-47)
    folder_index += last_folder
    print "Part " + str((folder_index+1)) + " of 48"
    polygon_path = SUB_DIRECTORIES[folder_index] + "/polygons.mat"
    polygon_file = clean_polygon_mat(polygon_path)
    for image_index, image_path in enumerate(folder):
      tf_example = create_tf_example(polygon_file, image_path , image_index)
      writer.write(tf_example.SerializeToString())
    last_folder1 += 1 #doing the same counting thing for the next loop
  writer.close()

  #for eval
  print "working on the eval split"
  write_path = cwd + "/DatasetTFRecords/eval.record"
  writer = tf.python_io.TFRecordWriter(write_path)
  for folder_index, folder in enumerate(eval_sample_files):
    #since we've taken folders 46 and 47 we need to add 46 to the folder_index to get us to the right place
    #however a better solution is to count the number of folders we've been through in the last loop 
    #this means we can change the splits without having to come back down here and worry about the increments (lines 45-47)
    folder_index += last_folder1
    print "Part " + str((folder_index+1)) + " of 48"
    polygon_path = SUB_DIRECTORIES[folder_index] + "/polygons.mat"
    polygon_file = clean_polygon_mat(polygon_path)
    for image_index, image_path in enumerate(folder):
      tf_example = create_tf_example(polygon_file, image_path , image_index)
      writer.write(tf_example.SerializeToString())
  writer.close()

if __name__ == '__main__':
  tf.app.run()