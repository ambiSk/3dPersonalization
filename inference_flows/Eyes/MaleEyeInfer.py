import pandas as pd
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import dlib
from imutils import face_utils
from fastai.vision.all import *
from fastai.metrics import error_rate
import os
import torch
import pickle
import glob
import sys


# sys.path.append("/nfs_storage/fs-mnt6/pranshu/selfie_sticker/fastai_training/MTL_V4_1_rerun_new_data/hifai/")
# sys.path.append("/nfs_storage/fs-mnt6/pranshu/selfie_sticker/fastai_training/MTL_V4_1_rerun_new_data")
# import hifai
# from mtl.utils import NanLabelImageList
# from mtl.normalizer import NormalizedFloatList
# from mtl.multitask import label_from_mt_project
# from mtl.multitask import MultitaskMetrics

# json_dir = "../Male_output_json/"
MODEL_PATH = "./Trained_models/Eyes/Male/renet18_male_eyes.pkl"
TASK_OUTPUT_MAPPER = 'Hikemoji3D/Assets/PythonScripts/Trained_models/Eyes/Male/task_output_mapper.pkl'
FACE_LANDMARKS_FILE = "/home/sharathchandra/shape_predictor_68_face_landmarks.dat"

# def ModelInfer(image_dir, MODEL_PATH, TASK_OUTPUT_MAPPER):
#     learn = load_learner(MODEL_PATH, test=ImageList.from_df(path=image_dir, df=pd.DataFrame(os.listdir(image_dir), columns=["ImagePath"])))

#     preds, _ = learn.get_preds(ds_type=DatasetType.Test)

#     expected_cols = [(k,v) for (k,vs) in pickle.load(open(TASK_OUTPUT_MAPPER, 'rb')).items() for v in vs]
#     assert len(expected_cols)==preds.shape[1]

#     preds = pd.DataFrame(np.array(preds), columns=pd.MultiIndex.from_tuples(expected_cols))
#     Eye_cols = [col for col in preds.columns if col[0]==("MaleEyes")]
#     preds = preds[Eye_cols]

#     indextocol_dict = {}
#     for i,col in enumerate(Eye_cols):
#         indextocol_dict[i+1] = col[1]

#     max_args = np.argmax(np.array(preds),axis=-1,)

#     eye_type = []
#     for i in max_args:
#         eye_type.append(indextocol_dict[i])

#     df = pd.concat([pd.DataFrame(os.listdir(INPUT_PATH), columns=["ImagePath"]),pd.DataFrame(eye_type, columns=["EyeType"])],axis=1)

#     return df

def ModelInfer(image_dir, MODEL_PATH):
    learn = load_learner(MODEL_PATH)

    df = pd.DataFrame(os.listdir(image_dir), columns=["ImagePath"])
    df['EyeType'] = df['ImagePath'].apply(lambda x: learn.predict(image_dir+x)[0] if (x.endswith('jpg') or x.endswith('jpeg') or x.endswith('png')) else None)
    return df


def MaleEyeInfer(image_dir):
    male_eye_dict = {}
    male_eye_dict['ImagePath'] = []
    male_eye_dict['EyeType'] = []

    dict_min_max = {}
    dict_min_max['bridge_eye_dist_normalized_min'],dict_min_max['bridge_eye_dist_normalized_max'] = 0.1849746069979616, 0.2522663243994686

    # df = ModelInfer(image_dir, MODEL_PATH)

    dict_with_params = {}
    dict_with_params['ImagePath'] = []
    dict_with_params['face_width'] = []
    dict_with_params['bridge_eye_dist'] = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACE_LANDMARKS_FILE)

    cnt = 0
    for file in os.listdir(image_dir):
        if file!='output_stickers' and file!='output_json':
            try:
                img_path = image_dir+file
                image=cv2.imread(img_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)

                x_coord = []
                y_coord = []
                final_array = []
                preds = []
                for rect in rects:
                    pred=predictor(gray,rect)
                    preds.append(pred)
                    x_coord.extend(face_utils.shape_to_np(pred)[:,0])
                    y_coord.extend(face_utils.shape_to_np(pred)[:,1])  
                    final_array.extend(face_utils.shape_to_np(pred))

                if len(x_coord)>0:
                    nose_start = np.array((x_coord[27],y_coord[27]))

                    face_l = np.array((x_coord[36],y_coord[36]))
                    face_r = np.array((x_coord[45],y_coord[45]))
                    face_width = np.linalg.norm(face_r-face_l)

                    p1 = np.array((x_coord[39],y_coord[39]))
                    p2 = np.array((x_coord[42],y_coord[42]))

                    bridge_to_eye1 = np.linalg.norm(nose_start-p1)
                    bridge_to_eye2 = np.linalg.norm(nose_start-p2)
                    bridge_to_eye = (bridge_to_eye1+bridge_to_eye2)/2

                    dict_with_params['ImagePath'].append(file)
                    dict_with_params['face_width'].append(face_width)
                    dict_with_params['bridge_eye_dist'].append(bridge_to_eye)
            
            except Exception as e:
                print(e)
                print(file)
                cnt+=1

    df_params = pd.DataFrame.from_dict(dict_with_params)
    df_params['bridge_eye_dist_normalized'] = df_params['bridge_eye_dist']/df_params['face_width']
    df_params['L_Eye_IN_Cor_Out'] = (df_params['bridge_eye_dist_normalized'] - dict_min_max['bridge_eye_dist_normalized_min'])/(dict_min_max['bridge_eye_dist_normalized_max'] - dict_min_max['bridge_eye_dist_normalized_min'])*30
    df_params['R_Eye_IN_Cor_Out'] = df_params['L_Eye_IN_Cor_Out']

    df_male_final = df_params[['ImagePath','L_Eye_IN_Cor_Out','R_Eye_IN_Cor_Out']]
    df_male_final.columns = ['image_name','L_Eye_IN_Cor_Out','R_Eye_IN_Cor_Out']

    # df_male_final['preset_Eyes'] = df_male_final['preset_Eyes'].apply(lambda x: mapping[x] if x in mapping else x)

    return df_male_final


