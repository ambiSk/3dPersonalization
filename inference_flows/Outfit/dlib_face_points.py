import dlib
import numpy as np
from gevent import lock
from imutils import face_utils
from PIL import Image
import logging


class DlibFacePointsDetector(object):
    _instance = None
    _lock = lock.RLock()

    def __init__(self):

        dlib_pretrained_model = "/home/sharathchandra/shape_predictor_68_face_landmarks.dat"
        self.dlib_face_detector = dlib.get_frontal_face_detector()
        self.dlib_face_points_predictor = dlib.shape_predictor(dlib_pretrained_model)

    @classmethod
    def get_instance(cls):
        if not DlibFacePointsDetector._instance:
            with cls._lock:
                if not DlibFacePointsDetector._instance:
                    DlibFacePointsDetector._instance = DlibFacePointsDetector()
        return DlibFacePointsDetector._instance

    def detect_face_bounding_box(self, image):
        face_bounding_boxes = self.detect_face_bounding_boxes_and_return_all(image)
        if not face_bounding_boxes and len(face_bounding_boxes) <= 0:
            return None
        biggest_bounding_box_index = self.get_biggest_bounding_box(face_bounding_boxes)
        if biggest_bounding_box_index is not None:
            return face_bounding_boxes[biggest_bounding_box_index]
        else:
            return None

    def detect_face_bounding_boxes_and_return_all(self, image):
        face_bounding_boxes = self.dlib_face_detector(image, 0)

        if not face_bounding_boxes:
            return []
        else:
            return face_bounding_boxes

    def detect_face_shape_points(self, image, bounding_box=None):
        if bounding_box is None:
            bounding_box = self.detect_face_bounding_box(image)
            if bounding_box is None:
                return None, None
        face_points = self.dlib_face_points_predictor(image, bounding_box)
        face_points_np = face_utils.shape_to_np(face_points)
        return face_points, face_points_np, bounding_box

    def detect_face_points(self, image, bounding_box=None):
        face_points, face_points_np, bounding_box = self.detect_face_shape_points(image, bounding_box)
        return face_points_np, bounding_box

    def get_biggest_bounding_box(self, bounding_boxes):
        index_of_bounding_box = -1
        max_area = -1

        for i, bounding_box_each in enumerate(bounding_boxes):
            height = bounding_box_each.height()
            width = bounding_box_each.width()
            area = height * width

            if area > max_area:
                max_area = area
                index_of_bounding_box = i

        if index_of_bounding_box == -1:
            return None
        else:
            return index_of_bounding_box

    def prewarm_model(self, image_path):
        image = Image.open(image_path)
        image = np.asarray(image, dtype=np.uint8)
        self.detect_face_shape_points(image)