import numpy as np
import cv2
cv2.setNumThreads(2)
from imutils import face_utils
from sklearn.cluster import KMeans
import imutils
import logging
from PIL import Image
import matplotlib.pyplot as plt

import os
import csv

# from ml_engine.color_extractor.color_extractor_assets import male_eyewear_with_color, female_eyewear_with_color
#
from inference_flows.Outfit.dlib_face_points import DlibFacePointsDetector

from inference_flows.Outfit.background_segmentation_v2 import BackgroundSegmentation


class ColorExtractor(object):
    __version = "v1"

    def get_version(cls):
        return cls.__version

    def __init__(self):

        self.face_points_left_eye_indices = face_utils.FACIAL_LANDMARKS_IDXS.get("left_eye")
        self.face_points_right_eye_indices = face_utils.FACIAL_LANDMARKS_IDXS.get("right_eye")
        self.face_points_nose_indices = face_utils.FACIAL_LANDMARKS_IDXS.get("nose")
        self.face_point_lips_indices = face_utils.FACIAL_LANDMARKS_IDXS.get("mouth")
        self.face_points_jaw_indices = face_utils.FACIAL_LANDMARKS_IDXS.get("jaw")

        self.lower_thresh = np.array([0, 48, 80], dtype="uint8")
        self.upper_thresh = np.array([20, 255, 255], dtype="uint8")

        self.default_hair_color_male = "#381D1D"
        self.default_hair_color_female = "#381D1D"

        self.default_facial_hair_color_male = "#381D1D"

        self.default_body_color_male = "#333333"
        self.default_body_color_female = "#FF6699"

        self.default_lip_color_male = "#FACBA9"
        self.default_lip_color_female = "#ED9494"

        self.default_headgear_color_male = "#0000FF"
        self.default_headgear_color_female = "#0000FF"

        self.default_sunglass_color_male = "#696969"
        self.default_sunglass_color_female = "#696969"

        self.default_skin_color_male = "#FFB799"
        self.default_skin_color_female = "#FFB799"

        # self.male_eyewear_with_color = male_eyewear_with_color
        # self.female_eyewear_with_color = female_eyewear_with_color

        # self.lip_color_list = ["EA245F", "911365", "D57A8B", "CB454E", "F14287",
        #                        "D9152B", "DC2E85", "A35F60","C7568C", "F64847",
        #                        "E851BC","E267CF", "F4494F", "870E1F", "A92E33",
        #                        "CB446C", "C86C81","E11657", "D1216A", "B75B0D",
        #                        "B8001F", "BC0D68", "AA2B38", "AA2B0A"]

        self.lip_color_list = ["EA245F", #"911365",
                               # "D57A8B",
                               "CB454E", "F14287",
                               "D9152B", "DC2E85", "C7568C", "F64847",
                               "E851BC", "E267B9", "F4494F", "B72450",
                               # "870E1F",
                               # "9F1326", Can use
                               # #"A92E33",
                               "CB446C", "C86C81","E11657", "D1216A",
                               #"B8001F",
                               "BC0D68", #"AA2B38",
                               "FF69B4", "FF1493", "DB7093", "C71585", "FF00FF", "F08080", "DC143C",# named colors
                               "FF94A7", "FF83C1", "FF8695", "FF6489", "FF6C81",
                               # shades of red color
                               "fb607f", "ff4040", "ff355e", "ef3038", "ed2939", "cb4154", "e32636", "da2c43", "e62020",
                               "ff033e" , "e51a4c", "ff003f", "f2003c", "d9004c",
                               "C65067", "E05F8C", "D53155", "B72438", "D26689", "CF6385",  #"A95171",
                               "BC5C80", "B45171"
                               ]

        self.lip_points_start_range = 48
        self.lip_points_end_range = 60
        self.lip_resize_shape = (200, 100)

    def check_key_and_value_for_existance(self, component, component_dictionary, face_points):
        if component_dictionary.get(component, {}).get("value", "None") != 'None' and face_points is not None:
            return True
        else:
            return False

    def _check_color_filling_male(self, component, component_dictionary, face_points):

        if component in ["MaleLips", "MaleEyewear", "MaleHairFront", "MaleBodyShapes"]:
            return self.check_key_and_value_for_existance(component, component_dictionary, face_points)

        logging.warn("Component name was did not match with any known component")
        return False

    def _check_color_filling_female(self, component, component_dictionary, face_points):

        if component in ["FemaleLips", "FemaleEyewear", "FemaleHairFront", "FemaleBodyShapes"]:
            return self.check_key_and_value_for_existance(component, component_dictionary, face_points)

        logging.warn("Component name did not match with any known component")
        return False

    def check_color_filling(self, gender, component, component_dictionary, face_points):

        if gender == "male":
            return self._check_color_filling_male(component, component_dictionary, face_points)
        else:
            return self._check_color_filling_female(component, component_dictionary, face_points)

    def get_dominant_color_hex(self, image):
        color_list = self.remove_black_white_points(image)
        color_rgb = self.get_dominant_cluster_color(color_list)
        if color_rgb is None:
            return None

        color_rgb = np.floor(color_rgb)
        color_hex = self.convert_rgb2hex(color_rgb)
        return color_hex

    def get_dominant_cluster_color(self, image_color_rgb_list):

        if len(image_color_rgb_list) < 3:
            return None

        cluster = KMeans(n_clusters=3, n_init=3)
        cluster.fit(image_color_rgb_list)

        cluster_centers = cluster.cluster_centers_
        cluster_labels = cluster.labels_

        max_cluster_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_dominant_color_rgb = cluster_centers[np.argmax(counts)]

        return cluster_dominant_color_rgb

    def retain_skin(self, body_part_image):
        converted = cv2.cvtColor(body_part_image, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, self.lower_thresh, self.upper_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

        skin = cv2.bitwise_and(body_part_image, body_part_image, mask=skinMask)

        return skin

    def remove_skin(self, body_part_image):
        converted = cv2.cvtColor(body_part_image, cv2.COLOR_RGB2HSV)
        skinMask = cv2.inRange(converted, self.lower_thresh, self.upper_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skinMask = cv2.bitwise_not(skinMask)

        skin = cv2.bitwise_and(body_part_image, body_part_image, mask=skinMask)

        return skin

    def increase_by_delta(self, x, y, w, h, delta):
        w = int(np.round(w * (1 + (delta / 100))))
        h = int(np.round(h * (1 + (delta / 100))))
        return (x, y, w, h)

    def convert_color_RGB2BGR(self, color):
        color = np.array(color, np.uint8)
        color = color.reshape((1, 1, -1))
        new_color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        new_color = new_color.reshape((-1))
        return new_color

    def convert_color_BGR2RGB(self, color):
        color = np.array(color, np.uint8)
        color = color.reshape((1, 1, -1))
        new_color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        new_color = new_color.reshape((-1))
        return new_color

    def convert_rgb2hex(self, color):

        hex_color = ""
        for i in range(len(color)):
            if color[i] < 16:
                hex_color += '0' + hex(int(color[i])).upper().lstrip("0X")
            else:
                hex_color += hex(int(color[i])).upper().lstrip("0X")

        return "#" + hex_color

    def remove_black_white_points(self, image):

        dummy_image = image.copy()

        mask = np.logical_and.reduce((dummy_image[:, :, 0] > 2, dummy_image[:, :, 1] > 2, dummy_image[:, :, 2] > 2,
                                      dummy_image[:, :, 0] < 250, dummy_image[:, :, 1] < 250,
                                      dummy_image[:, :, 2] < 250))

        return np.array(dummy_image[mask])

    def remove_face(self, image, face_points):

        try:
            clone = image.copy()

            y_index_above_left_eyebrow = face_points[19][1]
            y_index_above_right_eyebrow = face_points[24][1]

            x_index_left_jaw = face_points[0][0]
            x_index_right_jaw = face_points[16][0]

            image_after_face_removal = clone[0:min(y_index_above_left_eyebrow, y_index_above_right_eyebrow),
                                       x_index_left_jaw:x_index_right_jaw]

            if image_after_face_removal is not None and image_after_face_removal.size !=0:
                image_after_face_removal = cv2.resize(image_after_face_removal, (50, 50), cv2.INTER_AREA)
            else:
                return None

            return image_after_face_removal

        except Exception:
            logging.exception("cannot remove face from image")
            return None

    def below_face(self, image_background_segmented, face_points):
        try:
            clone = image_background_segmented.copy()
            y_face_bottom_point = max(face_points[:, 1])
            image_after_face_removal = clone[y_face_bottom_point:, :]

            if image_after_face_removal is not None and image_after_face_removal.size !=0:
                image_after_face_removal = cv2.resize(image_after_face_removal, (50,50), cv2.INTER_AREA)
            else:
                return None

            return image_after_face_removal

        except Exception as e:
            logging.exception("body segmentation for body color not found")
            return None

    def remove_t_points(self, image, face_points):

        try:
            clone = image.copy()

            left_eye_contours = [
                np.array(face_points[self.face_points_left_eye_indices[0]:self.face_points_left_eye_indices[1]],
                         dtype=np.int32)]
            right_eye_contours = [
                np.array(face_points[self.face_points_right_eye_indices[0]:self.face_points_right_eye_indices[1]],
                         dtype=np.int32)]
            lips_contours = [
                np.array(face_points[48:60], dtype=np.int32)]
            jaw_contours = [
                np.array(face_points[self.face_points_jaw_indices[0]:self.face_points_jaw_indices[1]], dtype=np.int32)]

            mask = np.zeros((image.shape[0], image.shape[1]))

            cv2.fillPoly(clone, pts=left_eye_contours, color=0)
            cv2.fillPoly(clone, pts=right_eye_contours, color=0)
            cv2.fillPoly(clone, pts=lips_contours, color=0)

            cv2.fillPoly(mask, pts=jaw_contours, color=1)
            mask = mask.astype(np.bool)

            image_after_t_removal = np.zeros_like(clone)

            try:
                image_after_t_removal[mask] = clone[mask]
            except Exception:
                logging.exception("cannot find mask or cannot overlay mask")
                return None

            return image_after_t_removal

        except Exception:
            logging.exception("cannot remove t points in image")
            return None

    def eye_points_extraction(self, image, face_points):

        try:
            clone = image.copy()

            left_eye_contours = [
                np.array(face_points[self.face_points_left_eye_indices[0]:self.face_points_left_eye_indices[1]],
                         dtype=np.int32)]
            right_eye_contours = [
                np.array(face_points[self.face_points_right_eye_indices[0]:self.face_points_right_eye_indices[1]],
                         dtype=np.int32)]

            mask = np.zeros((clone.shape[0], clone.shape[1]))

            cv2.fillPoly(mask, pts=left_eye_contours, color=1)
            cv2.fillPoly(mask, pts=right_eye_contours, color=1)
            mask = mask.astype(np.bool)

            eye_points = np.zeros_like(clone)

            eye_points[mask] = clone[mask]

            if eye_points is not None and eye_points.size !=0:
                eye_points = cv2.resize(eye_points, (50,50) ,cv2.INTER_AREA)
            else:
                return None

            return eye_points

        except Exception as e:
            logging.exception("cannot extract eye points from image")
            return None

    def find_face_points(self, image):

        try:
            clone_image = image.copy()
            gray_image = cv2.cvtColor(clone_image, cv2.COLOR_RGB2GRAY)

            bounding_box = DlibFacePointsDetector.get_instance().detect_face_bounding_box(gray_image)
            if not bounding_box:
                logging.warn("Could not find bounding box using dlib")
                return None

            face_points, bounding_box = DlibFacePointsDetector.get_instance().detect_face_points(gray_image, bounding_box)
            return face_points
        except Exception:
            logging.exception("dlib failed to detect face/points in preprocess_image in ColorExtractor")
            return None

    def get_colors(self, image_background_segmented, bg_seg_face_points, gender, timer_prefix=None):

        component_colors_found = {}

        if image_background_segmented is None or bg_seg_face_points is None:
            return

        if gender == "male":

            eyewear_color = self.extract_sunglass_color(image_background_segmented, bg_seg_face_points)
            if eyewear_color is not None:
                component_colors_found["EyewearColor"] = eyewear_color

            # TODO: MODIDY HAIR COLOR FUNCTION TO BE MORE EXACT. RETURNING DEFAULT HAIR COLOR FOR NOW.
            # hair_color = self.extract_hair_color(image_background_segmented, bg_seg_face_points)
            hair_color = None
            if hair_color is not None:
                component_colors_found["HairColor"] = hair_color

            body_color = self.extract_body_color(image_background_segmented, bg_seg_face_points)
            if body_color is not None:
                component_colors_found["OutfitColor"] = body_color

            headgear_color = self.extract_headgear_color(image_background_segmented, bg_seg_face_points)
            if headgear_color is not None:
                component_colors_found["HeadgearColor"] = headgear_color

            lip_color = self.extract_lip_color(image_background_segmented, bg_seg_face_points)
            # lip_color = None
            if lip_color is not None:
                component_colors_found["LipColor"] = lip_color

        else:

            eyewear_color = self.extract_sunglass_color(image_background_segmented, bg_seg_face_points)
            if eyewear_color is not None:
                component_colors_found["EyewearColor"] = eyewear_color

            # TODO: SWTICH THIS ON WITH LESS DARK LIP COLORS
            lip_color = self.extract_lip_color(image_background_segmented, bg_seg_face_points)
            # lip_color = None
            if lip_color is not None:
                component_colors_found["LipColor"] = lip_color

            # TODO: MODIDY HAIR COLOR FUNCTION TO BE MORE EXACT. RETURNING DEFAULT HAIR COLOR FOR NOW.
            # hair_color = self.extract_hair_color(image_background_segmented, bg_seg_face_points)
            hair_color = None
            if hair_color is not None:
                component_colors_found["HairColor"] = hair_color

            body_color = self.extract_body_color(image_background_segmented, bg_seg_face_points)
            if body_color is not None:
                component_colors_found["OutfitColor"] = body_color

            headgear_color = self.extract_headgear_color(image_background_segmented, bg_seg_face_points)
            if headgear_color is not None:
                component_colors_found["HeadgearColor"] = headgear_color

        return component_colors_found

    def extract_sunglass_color(self, image, face_points):

        eye_points_image = self.eye_points_extraction(image, face_points)

        if eye_points_image is None:
            return None

        skin_removed_image = self.remove_skin(eye_points_image)

        return self.get_dominant_color_hex(skin_removed_image)

    def extract_hair_color(self, image, face_points):

        face_removed_image = self.remove_face(image, face_points)
        if face_removed_image is None:
            return None

        skin_removed_image = self.remove_skin(face_removed_image)

        return self.get_dominant_color_hex(skin_removed_image)

    def extract_headgear_color(self, image, face_points):

        face_removed_image = self.remove_face(image, face_points)
        if face_removed_image is None:
            return None

        skin_removed_image = self.remove_skin(face_removed_image)

        return self.get_dominant_color_hex(skin_removed_image)

    def extract_body_color(self, image_background_segmented, face_points):

        image_below_face = self.below_face(image_background_segmented, face_points)

        if image_below_face is None or image_below_face.size == 0:
            logging.error("Image below face could not be extracted out in body color.")
            return None

        skin_removed_image = self.remove_skin(image_below_face)

        return self.get_dominant_color_hex(skin_removed_image)

    def extract_skin_color(self, image, face_points):

        t_section_removed_image = self.remove_t_points(image, face_points)

        if t_section_removed_image is None:
            return None

        skin_parts = self.retain_skin(t_section_removed_image)

        return self.get_dominant_color_hex(skin_parts)

    def extract_lip_color(self, image, face_points, delta=10):

        relevant_lips_points = face_points[self.lip_points_start_range:self.lip_points_end_range]
        contours = [np.array(relevant_lips_points, dtype=np.int32)]

        mask = np.zeros((image.shape[0], image.shape[1]))
        cv2.fillPoly(mask, pts=contours, color=1)
        mask = mask.astype(np.bool)

        lips = np.zeros_like(image)

        try:
            lips[mask] = image[mask]
        except Exception:
            logging.exception("cannot find mask or cannot overlay mask")
            return None

        try:
            (x, y, w, h) = cv2.boundingRect(
                np.array([face_points[self.face_point_lips_indices[0]:self.face_point_lips_indices[1]]]))

            (x_rs, y_rs, w_rs, h_rs) = self.increase_by_delta(x, y, w, h, delta)

            try:
                lips = lips[y_rs:y_rs + h_rs, x_rs:x_rs + w_rs]
            except:
                lips = lips[y:y + h, x:x + w]

        except Exception:
            logging.exception("cannot make bounding box")
            return None

        try:
            lips = cv2.resize(lips, self.lip_resize_shape)
        except Exception as e:
            logging.exception("cannot resize lips")

        skin_removed_lips = lips # self.remove_skin(lips)
        color_list = self.remove_black_white_points(skin_removed_lips)
        color_rgb = self.get_dominant_cluster_color(color_list)

        if color_rgb is None:
            return None

        color_rgb = np.floor(color_rgb)

        try:
            index_lip = 1
            count = 0
            min_dist = 1000000
            for p in self.lip_color_list:
                template_color = tuple(int(p[i:i + 2], 16) for i in (0, 2, 4))
                distance = max(abs(template_color[0] - color_rgb[0]), abs(template_color[1] - color_rgb[1]),
                               abs(template_color[2] - color_rgb[2]))
                if distance <= min_dist:
                    min_dist = distance
                    index_lip = count
                count = count + 1
            if min_dist < 70:
                color_hex = "#" + self.lip_color_list[index_lip]
                return color_hex
            else:
                return None
        except:
            logging.exception("Cannot calculate distance lip color, sending default.")
            return None

def run_inference(image_path):
    background_segmentation_obj = BackgroundSegmentation()
    color_extractor_obj = ColorExtractor()

    if not (image_path.endswith("png") or image_path.endswith("jpg") or image_path.endswith("jpeg")):
        return

    image = Image.open(image_path)
    resized_image, resized_image_size = background_segmentation_obj.preprocess_image(image)
    segmentation_map = background_segmentation_obj.run_inference(resized_image)
    bg_segmented_image = background_segmentation_obj.overlay_segmentation_on_image(resized_image, segmentation_map)

    resized_image = np.asarray(resized_image)
    face_points = color_extractor_obj.find_face_points(resized_image)

    lip_color = ColorExtractor().get_colors(bg_segmented_image, face_points, 'female').get("LipColor", None)
    
    if not lip_color:
        lip_color = "default"

    return {"lip_color":lip_color}



if __name__ == "__main__":
    image_path = "/Users/pankajdahiya/Downloads/female_raw_images/3c196c12-e922-4104-8b6a-794143e5f97eUoSJ_rkTDEYaNV7D.png"
    image = Image.open(image_path)

    bs = BackgroundSegmentation()
    resized_image, resized_image_size = bs.preprocess_image(image)
    segmentation_map = bs.run_inference(resized_image)
    bg_segmented_image =  bs.overlay_segmentation_on_image(resized_image, segmentation_map)

    cx = ColorExtractor()
    resized_image = np.asarray(resized_image)
    face_points = cx.find_face_points(resized_image)

    plt.imshow(resized_image)
    plt.show()

    print(ColorExtractor().get_colors(bg_segmented_image, face_points, 'female'))