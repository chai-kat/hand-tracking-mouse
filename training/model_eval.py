import os
import time

import cv2
import matplotlib
matplotlib.use("tkagg")

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
import tensorflow as tf
from scipy.io import loadmat



# ! Folder actions functions
def get_image_paths(folder_path):
    sorted_image_names = sorted(os.listdir(folder_path))

    output_path_list = []
    for name in sorted_image_names:
        if ".jpg" in name:
            output_path_list.append(os.path.join(folder_path, name))

    return output_path_list


def clean_polygon_mat(polygon_path):
    mat_dict = loadmat(polygon_path)
    mat = mat_dict["polygons"]
    mat = mat[0]
    mat = np.asarray(mat)
    return mat


# ! Get detection & truth box functions
def get_detections(frame, sess, graph):
    # need to do this to convert array from (h, w, 3) -> (1, h, w, 3);
    # the 1 here represents the amount of images we have
    frame_expanded = np.expand_dims(frame, axis=0)
    with graph.as_default():
        default_graph = tf.get_default_graph()
        det_boxes = default_graph.get_tensor_by_name("detection_boxes:0")
        det_scores = default_graph.get_tensor_by_name("detection_scores:0")
        img_tensor = default_graph.get_tensor_by_name('image_tensor:0')
        output_dict = sess.run([det_boxes, det_scores], feed_dict={
                               img_tensor: frame_expanded})

    y_dim, x_dim, channels = frame.shape

    boxes, scores = output_dict
    boxes = boxes[0]
    scores = scores[0]

    # de-normalize detections
    boxes[:, 0] = np.multiply(boxes[:, 0], y_dim)
    boxes[:, 1] = np.multiply(boxes[:, 1], x_dim)
    boxes[:, 2] = np.multiply(boxes[:, 2], y_dim)
    boxes[:, 3] = np.multiply(boxes[:, 3], x_dim)

    return boxes, scores


# ! Make-all-the-gotten-figures-into-shapely-objects functions
def get_truth_box(hand):
    points = shapely.geometry.MultiPoint(hand)
    xmin, ymin, xmax, ymax = points.bounds
    bb1 = shapely.geometry.box(xmin, ymin, xmax, ymax)

    return bb1


def get_prediction_box(bounding_box):
    xmin, ymin, xmax, ymax = bounding_box
    bb2 = shapely.geometry.box(xmin, ymin, xmax, ymax)

    return bb2


# ! Associate boxes function
def get_box_associations(boxes, image_annotations):
    # Make list of prediction + ground truth boxes
    gt_boxes_list = []
    gt_boxes_qty = 0

    for hand in image_annotations:
        if len(hand) > 1:
            gt_bb = get_truth_box(hand)
            gt_boxes_list.append(gt_bb)
            gt_boxes_qty += 1

    pr_boxes_list = []
    for box in boxes[:gt_boxes_qty]:
        pr_boxes_list.append(get_prediction_box(box))

    # Iterate over one box list and order the other by distance to it
    box_associations = []
    for gtb_index, gt_box in enumerate(gt_boxes_list):
        distances = []
        for pr_box in pr_boxes_list:
            dist = gt_box.distance(pr_box)
            distances.append(dist)
        # want to find the closest box
        closest_box_index = np.argmin(distances)
        closest_box = pr_boxes_list.pop(closest_box_index)

        association = (gt_box, closest_box)
        box_associations.append(association)

    return box_associations


# ! Display functions
def bb_rectangle(in_img, bb, is_gtruth, is_normalized=False):
    show_img = np.copy(in_img)
    xmin, ymin, xmax, ymax = bb
    

    if is_normalized:
        xmin = int(xmin * show_img.shape[1])
        ymin = int(ymin * show_img.shape[0])
        xmax = int(xmax * show_img.shape[1])
        ymax = int(ymax * show_img.shape[0])
    

    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    

    if not is_gtruth:
        # Predicted box is in red
        return cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=4) 
    else:
        # Ground truth box in green
        return cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=4)


if __name__ == "__main__":
    # For testing purposes
    graph_path = os.path.join("/Users/chaitanyakatpatal/Desktop/hand-tracking-mouse",
                              "hand_inference_graph/frozen_inference_graph.pb")
    folder_dir = "/Users/chaitanyakatpatal/Desktop/HandTrackingProduct/egohands_data/EvaluationSamples"
    # folder_dir = "" # ! You have to set this
    # graph_path = os.path.join(folder_dir, "hand_inference_graph/frozen_inference_graph.pb")
    evaluation_folders = ["PUZZLE_OFFICE_S_T", "PUZZLE_OFFICE_T_S"]

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session()

    for index, name in enumerate(evaluation_folders):
        evaluation_folders[index] = os.path.join(folder_dir, name)

    results = []  # A list of the IoUs of every hand in every image

    # Where the actual action happens
    for fpath in evaluation_folders:
        # * Get list of images, and polygon file for each folder.
        image_paths = get_image_paths(fpath)
        plygon_path = os.path.join(fpath, "polygons.mat")

        mat = clean_polygon_mat(plygon_path)

        for image_index, image_path in enumerate(image_paths):
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image_annotations = mat[image_index]

            boxes, scores = get_detections(img, sess, detection_graph)
            box_associations = get_box_associations(boxes, image_annotations)

            # Create a copy so we don't go and edit the image data
            show_img = np.copy(img)

            for association in box_associations:
                intersection = association[0].intersection(association[1]).area
                union = association[0].union(association[1]).area
                iou = intersection/union

                results.append(iou)

                show_img = bb_rectangle(show_img, association[0].bounds, is_gtruth=True)
                show_img = bb_rectangle(show_img, association[1].bounds, is_gtruth=False)

                msg = "Intersection: {}, Union: {}, IoU: {}".format(
                    intersection, union, iou)
                print msg
            
            plt.close()
            plt.imshow(show_img)
            plt.title('Don''t touch! I will proceed automatically.')

            plt.show(block=False)
            duration = 2    # [sec]
            plt.pause(duration)
            