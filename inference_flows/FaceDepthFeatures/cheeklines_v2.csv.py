import os
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
#%matplotlib inline

import csv
import shutil

from fastai.vision.all import *
import torch
from PIL import Image, ImageDraw, ImageFont

import mediapipe as mp


os.environ["CUDA_VISIBLE_DEVICES"] = ""

class CropImage:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                         min_detection_confidence=0.5)

    def get_coords(self, landmark, height, width):
        x1 = round(landmark[207].x * width)
        x2 = round(landmark[427].x * width)

        y1 = round(landmark[197].y * height)
        y2 = round(landmark[200].y * height)

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

model_path = "../../Trained_models/cheeklines_v2.pkl"
if not os.path.exists(model_path):
    os.system("gsutil cp gs://ds-staging-bucket/3D-Hikemoji-pers/trained-models/cheeklines_v2.pkl " + model_path)

learn = load_learner(model_path)


def run_inference(images_path, csv_file=None):
    tmp_path = "/tmp/tmp-dir-infer"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.mkdir(tmp_path)

    if os.path.isfile(images_path):
        img_paths = [images_path, ]
    else:
        img_paths = [os.path.join(images_path, img_path) for img_path in os.listdir(images_path)]

    if csv_file is not None:
        os.makedirs(os.path.join(csv_file.split("/")[-1]), exist_ok=True)
        out_file = open(csv_file, "w")
        csvwriter = csv.writer(out_file)
        csvwriter.writerow([ "image_name", "cheeklines"])

    for src_ in img_paths:
        if not (src_.endswith("png") or src_.endswith("jpg") or src_.endswith("jpeg")):
            continue
        tmp_dest = os.path.join(tmp_path, src_.split("/")[-1])

        crop_image_obj.get_cropped_image(src_, tmp_dest)
        pred = learn.predict(tmp_dest)[1].bool().numpy()[0]

        if csv_file is not None:
            csvwriter.writerow([src_.split("/")[-1], str(pred)])

        print("Done", src_)

    if csv_file:
        out_file.close()

    shutil.rmtree(tmp_path, ignore_errors=True)
