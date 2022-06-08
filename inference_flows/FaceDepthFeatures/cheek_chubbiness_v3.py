import os
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
#%matplotlib inline

import shutil

from fastai.vision.all import *
import torch
from PIL import Image, ImageDraw, ImageFont

import mediapipe as mp


os.environ["CUDA_VISIBLE_DEVICES"] = ""

# BS_VALUES_M = [("L_Cheek_Out", 60), ("R_Cheek_Out", 60), ("L_Cheek_Up", -30), ("R_Cheek_Up", -30),
#                ("Upper_Cheek_In_Out", -20), ("Upper_Cheek_Up_Dn", -20), ("Upper_Cheek_Fr_Bk", 60),
#                ("Lower_Cheek_Fr_Bk", 50), ("L_Cheek_Fr", 60), ("R_Cheek_Bk", -60),
#              ("L_Cheekbone_Out", 50), ("R_Cheekbone_Out", 50), ("L_Cheekbone_Up", -50), ("R_Cheekbone_Up", -50),
#              ("L_Cheekbone_Fr", 60), ("R_Cheekbone_Bk", 60), ("Lower_Cheek_In_Out", 35)]

BS_VALUES_M = [("L_Cheek_Out", 80), ("R_Cheek_Out", 80), ("L_Cheek_Up", -30), ("R_Cheek_Up", -30),
               ("Upper_Cheek_In_Out", -20), ("Upper_Cheek_Up_Dn", -20), ("Upper_Cheek_Fr_Bk", 50),
               ("Lower_Cheek_Fr_Bk", 60), ("L_Cheek_Fr", 60), ("R_Cheek_Bk", -60),
             ("L_Cheekbone_Out", 70), ("R_Cheekbone_Out", 70), ("L_Cheekbone_Up", -50), ("R_Cheekbone_Up", -50),
             ("L_Cheekbone_Fr", 60), ("R_Cheekbone_Bk", 60), ("Lower_Cheek_In_Out", 35), ("Lower_Cheek_Up_Dn", -50)]

UPPER_CHEEK_FR_BK_INDEX = 6

# BS_VALUES_M = [("L_Cheek_Out", 150), ("R_Cheek_Out", 150), ("L_Cheek_Up", -30),
#              ("R_Cheek_Up", -30), ( "Upper_Cheek_In_Out", -20), ( "Upper_Cheek_Up_Dn", -20),
#              ("Upper_Cheek_Fr_Bk", 60), ("Lower_Cheek_Fr_Bk", 100), ("L_Cheek_Fr", 90), ("R_Cheek_Bk", -90),
#              ( "L_Cheekbone_Out", 150), ("R_Cheekbone_Out", 150),
#              ("L_Cheekbone_Up", -150), ("R_Cheekbone_Up", -150),
#              ("L_Cheekbone_Fr", 100), ("R_Cheekbone_Bk",  -100), ("Lower_Cheek_In_Out", 75), ("Lower_Cheek_Up_Dn", 75)
#             ]

BS_VALUES_F = [("L_Cheek_Out", 60), ("R_Cheek_Out", 60), ("L_Cheek_Up", -30), ("R_Cheek_Up", -30),
               ("Upper_Cheek_In_Out", -20), ("Upper_Cheek_Up_Dn", -20), ("Upper_Cheek_Fr_Bk", 60),
               ("Lower_Cheek_Fr_Bk", 50), ("L_Cheek_Fr", 60), ("R_Cheek_Bk", -60),
             ("L_Cheekbone_Out", 30), ("R_Cheekbone_Out", 30), ("L_Cheekbone_Up", -50), ("R_Cheekbone_Up", -50),
             ("L_Cheekbone_Fr", 60), ("R_Cheekbone_Bk", 60), ("Lower_Cheek_In_Out", 35)]

class CropImage:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                         min_detection_confidence=0.5)

    def get_coords(self, landmark, height, width):
        x1 = min(round(landmark[234].x * width), round(landmark[127].x * width), round(landmark[93].x * width))
        x2 = max(round(landmark[454].x * width), round(landmark[356].x * width), round(landmark[323].x * width))

        y1 = min(round(landmark[168].y * height), round(landmark[193].y * height), round(landmark[417].y * height))
        y2 = max(round(landmark[152].y * height), round(landmark[148].y * height), round(landmark[377].y * height))

        return x1, x2, y1, y2

    def get_cropped_image(self, img_in, img_out):
        image = cv2.cvtColor(cv2.imread(img_in), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            cv2.imwrite(img_out, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) # Writing un-cropped image
            return

        for face_landmarks in results.multi_face_landmarks:
            (x1, x2, y1, y2) = self.get_coords(face_landmarks.landmark, height, width)
            image = image[y1: y2, x1: x2, ]

            cv2.imwrite(img_out, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            return

    def plot_image(self, img_in):
        img = mpimg.imread(img_in)
        plt.imshow(img)
        plt.show()


crop_image_obj = CropImage()

model_path = "../../Trained_models/cheek_chubbiness_v3.pkl"
if not os.path.exists(model_path):
    os.system("gsutil cp gs://ds-staging-bucket/3D-Hikemoji-pers/trained-models/cheek_chubbiness_v3.pkl " + model_path)

learn = load_learner(model_path)


def run_inference(image_path, gender='m'):
    tmp_path = "/tmp/ccv3"
    shutil.rmtree(tmp_path, ignore_errors=True)
    if not (image_path.endswith("png") or image_path.endswith("jpg") or image_path.endswith("jpeg")):
        return

    os.mkdir(tmp_path)
    tmp_dest = os.path.join(tmp_path, image_path.split("/")[-1])
    crop_image_obj.get_cropped_image(image_path, tmp_dest)
    pred = learn.predict(tmp_dest)[1].numpy()[0]
    pred = min(max(0, pred), 2)
    pred = round(pred * 2) / 4
    shutil.rmtree(tmp_path, ignore_errors=True)
    if gender=='m':
        curr_pred = [j*pred for _,j in BS_VALUES_M]
        curr_pred[UPPER_CHEEK_FR_BK_INDEX] -= 100
        return {BS_VALUES_M[i][0]:curr_pred[i] for i in range(len(curr_pred))}
        
    return {BS_VALUES_F[i][0]:BS_VALUES_F[i][1]*pred}

