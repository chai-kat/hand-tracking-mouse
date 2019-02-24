import argparse
import datetime
import sys
import time

import cv2
import numpy as np
import pyautogui
import tensorflow as tf

parser = argparse.ArgumentParser(
    description=""
)
parser.add_argument(
    "-c",
    "--camera",
    help="Camera device index to use",
    default=0,
    type=int
)
args = vars(parser.parse_args())
CAMERA_INDEX = args["camera"]

# Load frozen model into memory and make a new tf Session
F_PATH = "hand_inference_graph"
GRAPH_PATH = F_PATH + "/frozen_inference_graph.pb"
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(GRAPH_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session()


def get_centre(left, top, right, bottom):
    cx = (left+right)/2
    cy = (top+bottom)/2
    return cx, cy


def get_dark_percentage(thresh_inv):
    black_count = float(cv2.countNonZero(thresh_inv))
    img_area = thresh_inv.shape[0] * thresh_inv.shape[1]
    black_percentage = (black_count/img_area) * 100
    return black_percentage


# Function to get detections:
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
    return output_dict


# function to get image, in RGB
def get_img(cap, flip=False):
    # read the frame, discard the return code
    _, frame = cap.read()
    if flip:
        frame = cv2.flip(frame, 1)
    # cv2 uses BGR, tf uses RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, img_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img_thresh_inv = cv2.bitwise_not(img_thresh)
    return frame, img_thresh, img_thresh_inv


def get_mouse_points(nleft, ntop, nright, nbottom, s_height, s_width):
    # get centre
    cx, cy = get_centre(nleft, ntop, nright, nbottom)

    # get mouse points
    mx = cx * s_width
    my = cy * s_height
    # return mouse points
    return mx, my


def get_points(box):
    top = bb[0]  # ymin
    left = bb[1]  # xmin
    bottom = bb[2]  # ymax
    right = bb[3]  # xmax
    return left, top, right, bottom


def get_fingers(thresh, thresh_val=127, idsubtract=0):
    if thresh_val > 255:
        thresh_val = 255
    elif thresh_val < 0:
        thresh_val = 0

    #  Do some thresholding here
    img = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)

    #  Get contours in image
    image, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #  Get largest contour
    areas = [cv2.contourArea(c) for c in contours]
    try:
        cnt = contours[(np.argmax(areas)-idsubtract)]
    except ValueError:
        return None, thresh

    #  Get number of convexity defects (fingers)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    # TODO: NEED CODE TO ACTUALLY COUNT THE FINGERS
    # TODO: only to show the image, delete in the actual thing
    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    try:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(img, start, end, [0, 255, 0], 2)
            cv2.circle(img, far, 5, [0, 0, 255], -1)
    except:
        return None, img
    # second value is the display image
    return defects, img


def img_select(img, left, top, right, bottom, add_factor=0, normalized=False):
    height = img.shape[0]
    width = img.shape[1]

    if normalized:
        left = left*width
        right = right*width
        top = top*height
        bottom = bottom*height

    # ensure the points are integers and above 0
    left = int(left) - add_factor
    right = int(right) + add_factor
    top = int(top) - add_factor
    bottom = int(bottom) + add_factor

    if left < 0:
        left = 0
    if right > width:
        right = (width-1)
    if top < 0:
        top = 0
    if bottom > height:
        bottom = (height - 1)
    return img[top:bottom, left:right]


def move_mouse(mx, my):
    pyautogui.moveTo(mx, my)


def output(msg):
    sys.stdout.write('%s\r' % msg)
    sys.stdout.flush()


def prep_show_img(img, left, top, right, bottom):
    left = int(left * img.shape[1])
    top = int(top * img.shape[0])
    right = int(right * img.shape[1])
    bottom = int(bottom * img.shape[0])
    rectedImg = cv2.rectangle(img, (left, top), (right, bottom), [0, 255, 0], 3)
    showImg = cv2.cvtColor(rectedImg, cv2.COLOR_RGB2BGR)
    return showImg


def prep_show_roi(img, left, top, right, bottom, add_factor=0):
    left = (int(left * img.shape[1]) - add_factor)
    top = (int(top * img.shape[0]) - add_factor)
    right = (int(right * img.shape[1]) + add_factor)
    bottom = (int(bottom * img.shape[0]) + add_factor)
    relevant_frame = img_select(img, left, top, right, bottom)
    showImg = cv2.cvtColor(relevant_frame, cv2.COLOR_RGB2BGR)
    return showImg


def write_time(tstart):
    tdiff = (datetime.datetime.now() - tstart).total_seconds()
    msg = "Frame time: " + str(tdiff) + " seconds"
    output(msg)

def get_mouse_box(mx, my, box_side=400):
    xmin = mx - box_side
    ymin = my - box_side
    xmax = mx + box_side
    ymax = my + box_side
    return xmin, ymin, xmax, ymax


def is_in_mousebox(mx, my, mouse_box):
    xmin = mouse_box[0]
    ymin = mouse_box[1]
    xmax = mouse_box[2]
    ymax = mouse_box[3]

    if (mx >= xmin):
        if (mx <= xmax):
            if (my >= ymin):
                if (my <= ymax):
                    return True
    return False

screen_width, screen_height = pyautogui.size()
dark_frames = 0
mouse_box = None
mouse_in_box_frames = 0
thresh_val = 127

#  this seems usually to get to around 0.75 for an actual hand,
#  but it varies based on lighting, etc.
#  Change on the day of showcase
SCORE_THRESH = 0.6
# open VideoCapture and let the camera warm up
cap = cv2.VideoCapture(CAMERA_INDEX)
time.sleep(0.005)

while True:
    img, thresh, thresh_inv = get_img(cap, flip=True)
    boxes, scores = get_detections(img, sess, detection_graph)

    # failsafe, just cover the camera if the mouse goes out of control.
    if dark_frames == 10:
        break

    dpercentage = get_dark_percentage(thresh_inv)
    if (dpercentage > 95):
        dark_frames += 1
        continue
    else:
        if not dark_frames == 0:
            dark_frames -= 1

    # get only first box&score (highest probability):
    det_num = 0
    bb = boxes[0][det_num]
    score = scores[0][det_num]

    left, top, right, bottom = get_points(bb)

    if score >= SCORE_THRESH:
        mx, my = get_mouse_points(
            left, top, right, bottom, screen_height, screen_width)
        move_mouse(mx, my)

    else:
        mx, my = pyautogui.position()
    if mouse_box is None:
        bxmin, bymin, bxmax, bymax = get_mouse_box(mx, my, box_side=50)
        mouse_box = [bxmin, bymin, bxmax, bymax]
    else:
        if is_in_mousebox(mx, my, mouse_box):
            mouse_in_box_frames += 1
        else:
            bxmin, bymin, bxmax, bymax = get_mouse_box(mx, my, box_side=50)
            mouse_box = [bxmin, bymin, bxmax, bymax]

    if mouse_in_box_frames >= 4:
        pyautogui.click(pyautogui.position())
        mouse_in_box_frames = 0

    cv2.imshow("Captured Image + Detection",
               prep_show_img(img, left, top, right, bottom))
    if (cv2.waitKey(10) & 0xff) == ord("q"):
        cv2.destroyAllWindows()
        break

cap.release()
