import os
import cv2
import csv
import shutil
from joblib import load
import mediapipe as mp

POINTS_DICT = dict(
  silhouette= [
    454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234
  ],
  rightCheek= [205, 187, 147, 213, 192, 147, 123, 50, 137, 177, 245, 138, 135, 214],
  leftCheek= [425, 427, 411, 416, 433, 376, 280, 353, 366, 401, 435, 367, 364, 434]
)

POINTS_LIST = list()
for k, v in POINTS_DICT.items():
    POINTS_LIST.extend(v)
POINTS_SET = set(POINTS_LIST)

BS_VALUES = [("L_Cheek_Out", 85), ("R_Cheek_Out", 85), ("L_Cheek_Up", -80), ("R_Cheek_Up", -80),
             ("L_Cheekbone_Out", 70), ("R_Cheekbone_Out", 70), ("L_Cheekbone_Up", -100), ("R_Cheekbone_Up", -100),
             ("L_Cheekbone_Fr", 60), ("R_Cheekbone_Bk", 60), ("Upper_Cheek_Fr_Bk", -145), ("Lower_Cheek_In_Out", 35),
             ("Lower_Cheek_Fr_Bk", 75)]

class MpFeatures:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                         min_detection_confidence=0.5)

    def get_features(self, img_in):
        image = cv2.cvtColor(cv2.imread(img_in), cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            return None

        for face_landmarks in results.multi_face_landmarks:
            landmarks_x = [landmark.x for i, landmark in enumerate(face_landmarks.landmark) if i in POINTS_SET]
            landmarks_y = [landmark.y for i, landmark in enumerate(face_landmarks.landmark) if i in POINTS_SET]
            landmarks_z = [landmark.z for i, landmark in enumerate(face_landmarks.landmark) if i in POINTS_SET]

            landmarks = landmarks_x + landmarks_y + landmarks_z
            return landmarks


mp_features_obj = MpFeatures()

model_path = "../../Trained_models/cheek_chubbiness_v2.joblib"
if not os.path.exists(model_path):
    os.system("gsutil cp gs://ds-staging-bucket/3D-Hikemoji-pers/trained-models/cheek_chubbiness_v2.joblib " + model_path)

clf = load(model_path)


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
        csvwriter.writerow(["", "image_name"] + [i for i,j in BS_VALUES])

    for (i, src_) in enumerate(img_paths):
        if not (src_.endswith("png") or src_.endswith("jpg") or src_.endswith("jpeg")):
            continue
        # tmp_dest = os.path.join(tmp_path, src_.split("/")[-1])

        mp_features = mp_features_obj.get_features(src_)
        if mp_features is not None:
            pred = clf.predict_proba([mp_features,])[0]
        else:
            pred = 0

        if csv_file is not None:
            if pred[1] > 0.6:
                csvwriter.writerow([i, src_.split("/")[-1]] + [j for i,j in BS_VALUES])
            else:
                csvwriter.writerow([i, src_.split("/")[-1]] + [0] * len(BS_VALUES))

        print("Done", src_)

    if csv_file:
        out_file.close()

    shutil.rmtree(tmp_path, ignore_errors=True)
