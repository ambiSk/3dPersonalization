from inference_flows.Outfit.color_extractor import ColorExtractor
from inference_flows.Outfit.background_segmentation_v2 import BackgroundSegmentation

import numpy as np
import cv2
import torch
import albumentations as albu
import os
import csv
import dlib
from collections import namedtuple
from operator import mul
from functools import reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

from cloths_segmentation.pre_trained_models import create_model


model = create_model("Unet_2020-10-30")
model.eval()

color_extractor_obj = ColorExtractor()

predictor_path = "/home/sharathchandra/shape_predictor_68_face_landmarks.dat"

SPLINEPOINTS = 50

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

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

def get_landmarks(image):
    bounding_boxes_all = detector(image, 0)
    bounding_box_index = get_biggest_bounding_box(bounding_boxes_all)
    if bounding_box_index is None:
        return None
    bounding_box = bounding_boxes_all[bounding_box_index]
    face_points = predictor(image, bounding_box)
    return face_points

# def largest_rectangle(mask):
#     skip = 0
#     area_max = (0, [])
#
#     nrows, ncols = mask.shape
#     w = np.zeros(dtype=int, shape=mask.shape)
#     h = np.zeros(dtype=int, shape=mask.shape)
#     for r in range(nrows):
#         for c in range(ncols):
#             if mask[r][c] == skip:
#                 continue
#             if r == 0:
#                 h[r][c] = 1
#             else:
#                 h[r][c] = h[r - 1][c] + 1
#             if c == 0:
#                 w[r][c] = 1
#             else:
#                 w[r][c] = w[r][c - 1] + 1
#             minw = w[r][c]
#             for dh in range(h[r][c]):
#                 minw = min(minw, w[r - dh][c])
#                 area = (dh + 1) * minw
#                 if area > area_max[0]:
#                     area_max = (area, [(r - dh, c - minw + 1, r, c)])
#
#     for t in area_max[1]:
#         print('Cell 1:({}, {}) and Cell 2:({}, {})'.format(*t))
#     return area_max


class MaxRectangle:
    Info = namedtuple('Info', 'start height')

    @staticmethod
    def max_size(mat, value=1):
        """Find height, width of the largest rectangle containing all `value`'s."""
        it = iter(mat)
        hist = [(el==value) for el in next(it, [])]
        answer_row_end = 0
        max_size, answer_col_begin = MaxRectangle.max_rectangle_size(hist)

        for row_id, row in enumerate(it):
            hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]

            candidate_max_size, candidate_col_begin = MaxRectangle.max_rectangle_size(hist)
            if MaxRectangle.area(max_size) < MaxRectangle.area(candidate_max_size):
                answer_row_end = row_id
                answer_col_begin = candidate_col_begin
                max_size = candidate_max_size

        answer_row_end += 1

        return (answer_row_end - max_size[0] + 1,
                answer_row_end,
                answer_col_begin,
                answer_col_begin + max_size[1] - 1
                )
    @staticmethod
    def max_rectangle_size(histogram):
        """Find height, width of the largest rectangle that fits entirely under
        the histogram.
        """
        col_begin = 0
        stack = []
        top = lambda: stack[-1]
        max_size = (0, 0) # height, width of the largest rectangle
        pos = 0 # current position in the histogram
        for pos, height in enumerate(histogram):
            start = pos # position where rectangle starts
            while True:
                if not stack or height > top().height:
                    stack.append(MaxRectangle.Info(start, height)) # push
                elif stack and height < top().height:
                    # max_size = max(max_size, (top().height, (pos - top().start)),
                    #                key=area)
                    candidate_max_size = (top().height, (pos - top().start))
                    if MaxRectangle.area(max_size) <= MaxRectangle.area(candidate_max_size):
                        col_begin = top().start
                        max_size = candidate_max_size
                    # print("width", (pos - top().start))
                    start, _ = stack.pop()
                    continue
                break # height == top().height goes here

        pos += 1
        for start, height in stack:
            candidate_max_size = (height, (pos - start))
            if MaxRectangle.area(max_size) <= MaxRectangle.area(candidate_max_size):
                col_begin = top().start
        return max_size, col_begin

    @staticmethod
    def area(size):
        return reduce(mul, size)

def get_segmentation_output(image_path):
    # using background segmentation output
    image = Image.open(image_path)
    resized_image, _ = BackgroundSegmentation.get_instance().preprocess_image(image)
    segmentation_map = BackgroundSegmentation.get_instance().run_inference(resized_image)
    ####

    resized_image = np.array(resized_image)

    # image = load_rgb(image_path)
    # plt.imshow(image)
    # plt.show()

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(resized_image, factor=32, border=cv2.BORDER_CONSTANT)

    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)

    # using both masks
    mask = mask * (segmentation_map != 0)

    # using landmarks output
    landmark_points = get_landmarks(resized_image)
    if landmark_points:
        mask[: landmark_points.part(8).y, :] = 0

    largest_rectangle = MaxRectangle.max_size(mask)

    # masked_image = resized_image * mask[..., np.newaxis]
    masked_image = resized_image[largest_rectangle[0]: largest_rectangle[1], largest_rectangle[2]: largest_rectangle[3], :]

    plt.imshow(masked_image)
    plt.show()

    return masked_image


def run_inference(image_path, images_output_path=None):
    if not (image_path.endswith("png") or image_path.endswith("jpg") or image_path.endswith("jpeg")):
        return 

    if images_output_path is not None:
        os.makedirs(images_output_path, exist_ok=True)
    
    masked_image = get_segmentation_output(image_path)
    
    if images_output_path:
        cv2.imwrite(os.path.join(images_output_path, image_path.split("/")[-1]), cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

    dominant_color = color_extractor_obj.get_dominant_color_hex(masked_image)

    if not dominant_color:
        dominant_color = "default"
    
    return dict(outfit_color=dominant_color)
