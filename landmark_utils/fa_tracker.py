import face_alignment
import numpy as np
from typing import Tuple

class FaceAlignmentLandmarksTracker:

    def __init__(self, device='gpu'):
        if device == 'gpu':
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            print('Using gpu')
        else:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
        self.image = None
        self.landmarks = None

    def process(self, image):
        landmarks = self.fa.get_landmarks_from_image(image)
        if landmarks is not None:
            return landmarks[0]
        return None

    def set_image(self, image):
        self.image = image
        self.landmarks = self.process(image)

    def get_normalized_lip_thickness(self) -> Tuple[float, float]:
        if self.landmarks is None: return None

        eyes_mid = (self.landmarks[39] + self.landmarks[42]) / 2
        lip_mid = self.get_lip_mid()
        eyes_to_lip = np.linalg.norm(eyes_mid - lip_mid)

        upper_lip_normalized = self._get_upper_lip_thickness() / eyes_to_lip
        lower_lip_normalized = self._get_lower_lip_thickness() / eyes_to_lip
        return (upper_lip_normalized, lower_lip_normalized)

    def _get_upper_lip_thickness(self) -> float:
        upper_lip_right = (50, 61)
        upper_lip_mid = (51, 62)
        upper_lip_left = (52, 63)
        return (
                       np.linalg.norm(self.landmarks[upper_lip_right[0]] - self.landmarks[upper_lip_right[1]]) +
                       np.linalg.norm(self.landmarks[upper_lip_left[0]] - self.landmarks[upper_lip_left[1]])
               ) / 2

    def _get_lower_lip_thickness(self) -> float:
        lower_lip_right = (67, 58)
        lower_lip_mid = (66, 57)
        lower_lip_left = (65, 56)
        return (
                       np.linalg.norm(self.landmarks[lower_lip_right[0]] - self.landmarks[lower_lip_right[1]]) +
                       np.linalg.norm(self.landmarks[lower_lip_left[0]] - self.landmarks[lower_lip_left[1]])
               ) / 2

    def get_lip_mid(self):
        indices = [50, 61, 51, 62, 52, 63, 67, 58, 66, 57, 65, 56]
        relevant_landmarks = self.landmarks[indices]
        return relevant_landmarks.mean(axis=0)

    def get_lip_placement_v_ratio(self) -> float:
        ''' Returns ratio of
                lip to chin vs
                eyes to chin
        '''
        chin = self.landmarks[8]
        eyes_mid = (self.landmarks[39] + self.landmarks[42]) / 2
        lip_mid = self.get_lip_mid()
        nose = self.landmarks[33]
        return np.linalg.norm(lip_mid - chin) / np.linalg.norm(eyes_mid - chin)

    def get_left_lip_h_ratio(self) -> float:
        left_lip_landmark = self.landmarks[54]
        lip_mid = (self.landmarks[62] + self.landmarks[66]) / 2
        return np.linalg.norm(left_lip_landmark - lip_mid) / self.get_interocular_distance()

    def get_right_lip_h_ratio(self) -> float:
        right_lip_landmark = self.landmarks[48]
        lip_mid = (self.landmarks[62] + self.landmarks[66]) / 2
        return np.linalg.norm(right_lip_landmark - lip_mid) / self.get_interocular_distance()

    def get_interocular_distance(self) -> float:
        right_eye_inner = 39
        left_eye_inner = 42
        return np.linalg.norm(self.landmarks[left_eye_inner] - self.landmarks[right_eye_inner])

    def get_lips_crop(self):
        NOSE = 27
        CHIN = 8
        RIGHT_FACE = 3
        LEFT_FACE = 13
        nose = self.landmarks[NOSE]
        chin = self.landmarks[CHIN]
        right_face = self.landmarks[RIGHT_FACE]
        left_face = self.landmarks[LEFT_FACE]

        x_min = round(nose[1])
        y_min = round(right_face[0])
        x_max = round(chin[1])
        y_max = round(left_face[0])
        if x_min >= x_max or y_min >= y_max:
            raise Exception('Invalid crop of zero size')
        return self.image[x_min: x_max, y_min: y_max]

    def get_face_crop(self):
        x_min = round(self.landmarks[:, 1].min())
        x_max = round(self.landmarks[:, 1].max())
        y_min = round(self.landmarks[:, 0].min())
        y_max = round(self.landmarks[:, 0].max())
        if x_min >= x_max or y_min >= y_max:
            raise Exception('Invalid crop of zero size')
        return self.image[x_min: x_max, y_min: y_max]

    def get_lips_landmarks(self):
        if self.landmarks is None: return np.zeros((20, 2))
        return self.landmarks[48:68]
        
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path
    import cv2

    landmarks_tracker = FaceAlignmentLandmarksTracker(device='cpu')
    image_path = Path('/home/hike/Projects/data/Hikemoji3d/User_Data/images/ri_m_230_513_32c96c0f-7c8d-45df-9b48-805c8bbf7df6.png')
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks_tracker.set_image(image)
    lips_image = landmarks_tracker.get_lips_crop()

    plt.interactive(False)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(lips_image)

    plt.show(fig)
