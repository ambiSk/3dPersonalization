import pandas as pd
import dlib
import numpy as np

np.seterr(divide='ignore')

import pickle
import math
import argparse

import glob
import cv2
import dlib
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import ConvexHull, convex_hull_plot_2d, distance
import os
import shutil
import csv

from joblib import dump, load

import face_alignment
from skimage import io

# add the dlib path
predictor_path = "/home/sharathchandra/shape_predictor_68_face_landmarks.dat"

SPLINEPOINTS = 50

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def get_biggest_bounding_box(bounding_boxes):
    index_of_bounding_box = -1
    max_area = -1
    for i, bounding_box_each in enumerate(bounding_boxes):
        height = bounding_box_each.height()
        width = bounding_box_each.width()
        area = height*width
        if area > max_area:
            area = max_area
            index_of_bounding_box = i
    if index_of_bounding_box == -1:
        return None
    else:
        return index_of_bounding_box
def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def correct_pose(image):
    bounding_boxes_all = detector(image, 0)
    bounding_box_index = get_biggest_bounding_box(bounding_boxes_all)
    if bounding_box_index is None:
        return None
    bounding_box = bounding_boxes_all[bounding_box_index]
    face_points = predictor(image, bounding_box)
    left_eye = extract_left_eye_center(face_points)
    right_eye = extract_right_eye_center(face_points)
    M = get_rotation_matrix(left_eye, right_eye)
    w, h, c = image.shape
    image_max_size_possible = int(math.sqrt(w * w + h * h))
    offset_x = int((image_max_size_possible - image.shape[0]) / 2)
    offset_y = int((image_max_size_possible - image.shape[1]) / 2)
    dst_image = np.ones((image_max_size_possible, image_max_size_possible, 3), dtype='uint8') * 255
    dst_image[offset_x:(offset_x + image.shape[0]), offset_y:(offset_y + image.shape[1]), :] = image
    pose_corrected_image = cv2.warpAffine(src=dst_image, dsize=(image_max_size_possible, image_max_size_possible), M=M,
                                          borderValue=[255, 255, 255])
    return pose_corrected_image


def findface(image):
    bounding_boxes_all = detector(image, 1)
    bounding_box_index = get_biggest_bounding_box(bounding_boxes_all)
    if bounding_box_index is None:
        return None
    bounding_box = bounding_boxes_all[bounding_box_index]
    face_points = predictor(image, bounding_box)
    return face_points, bounding_box


def find_hull_im(im):
    shape, _bounding_box = findface(im)
    x = []
    y = []
    for i in range(2, 15):
        x.append(shape.part(i).x)
        y.append(shape.part(i).y)

    x = np.asarray(x)
    y = np.asarray(y)
    dist = np.sqrt((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2)
    dist_along = np.concatenate(([0], dist.cumsum()))

    spline, u = scipy.interpolate.splprep([x, y], u=dist_along, s=0)

    # resample it at smaller distance intervals
    interp_d = np.linspace(dist_along[0], dist_along[-1], SPLINEPOINTS)
    interp_x, interp_y = scipy.interpolate.splev(interp_d, spline)
    # plot(interp_x, interp_y, '-o')
    c = []
    for i in range(len(interp_x)):
        c.append([interp_x[i], interp_y[i]])
    c = np.asarray(c)

    hull1 = ConvexHull(c)
    return hull1, c, shape, _bounding_box


def normalize_points(np_array):
    max_point = np.max(np_array, axis=0)
    min_point = np.min(np_array, axis=0)
    height = max_point - min_point
    return (np_array - min_point) / height

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


import operator


def find_polyfit_coeff(X, Y, degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(X)

    model = LinearRegression(normalize=True)
    model.fit(x_poly, Y)
    # model.score(X, Y)
    y_poly_pred = model.predict(x_poly)

    #     plt.scatter(X, Y, s=10)
    #     # sort the values of x before line plot
    #     sort_axis = operator.itemgetter(0)
    #     sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
    #     x, y_poly_pred = zip(*sorted_zip)
    #     plt.plot(x, y_poly_pred, color='m')
    #     plt.show()
    return model.coef_


#     import operator
#     rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
#     r2 = r2_score(Y,y_poly_pred)
#     print(rmse)
#     print(r2)

def find_ratio(shape):
    x = []
    y = []
    for i in range(17):
        x.append(shape.part(i).x)
        y.append(shape.part(i).y)
    x = np.asarray(x)
    y = np.asarray(y)

    left = int(min(x))
    right = int(max(x))
    top = int(min(y))
    bottom = int(max(y))
    w = right - left
    h = bottom - top

    h = h + h / 1.61
    #     face_ratio_x = w/(h*1.5)
    face_ratio_x = h / w
    #     face_ratio_x = min(face_ratio_x,1.05)
    #     face_ratio_x = max(face_ratio_x,0.96)
    face_ratio_x = round(face_ratio_x, 2)

    return face_ratio_x


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((17, 2), dtype=dtype)
    for i in range(0, 17):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def find_angle_list_preset_face_shapes(face_dlib_points):
    list_angle = []
    angle_list = []
    for i in range(len(face_dlib_points) - 1):
        angle = (face_dlib_points[i][1] - face_dlib_points[i + 1][1]) / (
                    face_dlib_points[i][0] - face_dlib_points[i + 1][0])
        angle = np.arctan(angle)
        angle_list.append(angle)
    return angle_list


def get_best_preset_face_shape(face_points_shape):
    face_points_np = shape_to_np(face_points_shape)
    angle_list = find_angle_list_preset_face_shapes(face_points_np)
    return angle_list


def find_dist(p1, p2):
    return ((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)


class AlignedImage:
    def __init__(self):
        pass

    def get_aligned_image(self, img_in, img_out):
        try:
            image = cv2.cvtColor(cv2.imread(img_in), cv2.COLOR_BGR2RGB)
            # height, width = image.shape[:2]

            image = correct_pose(image)
            if image is None:
                return None

            cv2.imwrite(img_out, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return img_out

        except Exception as e:
            print(e)
            return None

    def plot_image(self, img_in):
        img = mpimg.imread(img_in)
        plt.imshow(img)
        plt.show()


class FacialLandmarks:
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    def get_landmarks(self, img_in):
        img_in = os.path.join(img_in)
        image = io.imread(img_in)
#         height, width = image.shape[:2]

        preds = self.fa.get_landmarks(image)

        if preds is None or len(preds) == 0:
            return None

        return preds[0]

    def get_relevant_features(self, img_in):

        landmark_points = self.get_landmarks(img_in)

        features = list()

        # features for lip width
        for (i, j) in [(50, 61), (52, 63), (67, 58), (66, 57), (65, 56), (61, 67), (62, 66), (63, 65)]:
            for (k, l) in [(49, 53), (33, 51), (27, 33), (27, 51), (33, 62)]:
                features.append(distance.euclidean(landmark_points[i], landmark_points[j])
                                / distance.euclidean(landmark_points[k], landmark_points[l]))

        # features for lip length
        for (i, j) in [(49, 53), (59, 55), (60, 64), (48, 54)]:
            for (k, l) in [(27, 8), (38, 43), (4, 12)]:
                features.append(distance.euclidean(landmark_points[i], landmark_points[j])
                                / distance.euclidean(landmark_points[k], landmark_points[l]))

        # gradient features
        for (i, j) in [(51, 55), (51, 59), (66, 58), (66, 56), (61, 66), (66, 63),
                       (49, 50), (59, 58), (52, 53), (55, 56), (49, 67), (53, 65),
                       (65, 55), (67, 59), (62, 67), (62, 65), (49, 66), (53, 66),
                       (60, 50), (52, 62), (60, 58), (64, 56)]:
            features.append(angle_between_2_points(landmark_points[i], landmark_points[j]))

        return features

    def save_landmark_image(self, img_in, landmark_points, save_landmarks_img):
        # os.makedirs(os.path.join(save_landmarks_img.split("/")[:-1]), exist_ok=True)

        img_in = cv2.cvtColor(cv2.imread(img_in), cv2.COLOR_BGR2RGB)

        CIRCLE_SIZE = 1
        THICKNESS_S = 1
        for landmark_point in landmark_points:
            cv2.circle(img_in, (int(landmark_point[0]), int(landmark_point[1])), CIRCLE_SIZE, color=(0, 0, 255), thickness=THICKNESS_S)

        cv2.imwrite(save_landmarks_img, cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR))


aligned_image_obj = AlignedImage()
facial_landmarks_obj = FacialLandmarks()

model_path = "Trained_models/Lips/female_lips_mouth_personalisation_xgb_v3.joblib"


lip_model = load(model_path)


BS_VALUES_M = [#("Up_Lip_Thick", 0), ("Low_Lip_Thick", 0), ("L_Lip_Con_Out", 0), ("R_Lip_Con_Out", 0),
               ("Mouth_Open", 0), ("L_Lip_Con_Up", 0), ("R_Lip_Con_Up", 0), ("M_Lip_Up", 0)]

BS_VALUES_F = [#("Up_Lip_Thick", 0), ("Low_Lip_Thick", 0), ("L_Lip_Con_Out", 0), ("R_Lip_Con_Out", 0),
               ("Mouth_Open", 0), ("L_Lip_Con_Up", 0), ("R_Lip_Con_Up", 0), ("M_Lip_Up", 0)]


def run_inference(image_path, gender='f'):
    tmp_path = "/tmp/LipsV3"
    shutil.rmtree(tmp_path, ignore_errors=True)
    if not (image_path.endswith("png") or image_path.endswith("jpg") or image_path.endswith("jpeg")):
        return
    os.mkdir(tmp_path)
    tmp_dest = os.path.join(tmp_path, image_path.split("/")[-1])
    curr_pred = None
    if aligned_image_obj.get_aligned_image(tmp_path, tmp_dest):
        input_features = facial_landmarks_obj.get_relevant_features(tmp_dest)

        if input_features:
            curr_pred = lip_model.predict([input_features, ])[0]
            curr_pred = [round(i * 100, 2) for i in curr_pred]
            curr_pred[3] = 0 if curr_pred[3] <= 3 else curr_pred[3]
    shutil.rmtree(tmp_path, ignore_errors=True)
    if gender == 'm':
        if curr_pred is None:
            return dict(BS_VALUES_M)
        else:
            curr_pred = [#curr_pred[0], curr_pred[1], curr_pred[2], curr_pred[2],
                            curr_pred[3], curr_pred[4], curr_pred[4], curr_pred[5]]

            return {BS_VALUES_M[i][0]:curr_pred[i] for i in range(len(curr_pred))}
            
    if gender == 'f':
        if curr_pred is None:
            return dict(BS_VALUES_F)
        else:
            curr_pred = [#curr_pred[0], curr_pred[1], curr_pred[2], curr_pred[2],
                            curr_pred[3], curr_pred[4], curr_pred[4], curr_pred[5]]
            return {BS_VALUES_F[i][0]:curr_pred[i] for i in range(len(curr_pred))}

    

    

