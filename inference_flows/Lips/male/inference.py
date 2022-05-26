import sys
from pathlib import Path
sys.path.append('../../..')
from landmark_utils.fa_tracker import FaceAlignmentLandmarksTracker

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from fastai.vision.all import *
from fastai.tabular.all import *
import torch


lip_columns = ["L_Lip_Con_Out", "L_Lip_Con_Up", "L_Lip_Con_Fr", "R_Lip_Con_Out", "R_Lip_Con_Up", "R_Lip_Con_Bk",
               "M_Lip_Up", "M_Lip_Fr", "Mouth_Open_BS", "Up_Lip_Thick", "Low_Lip_Thick", 
               "Up_Lip_02_IN_Out", "Up_Lip_02_Up_Dn", "Philtrum_Width"]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(len(lip_indices) * 2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, len(lip_columns)),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x) * 100


def get_blendshape_dict_from_learner(imagepath: Path, learner, tracker: FaceAlignmentLandmarksTracker):
    image = Image.open(imagepath).convert('RGB')
    image = np.asarray(image)
    tracker.set_image(image)
    lips_landmarks = tracker.get_lips_landmarks()
    lips_landmarks /= tracker.get_interocular_distance()
    normalized_lips_landmarks = lips_landmarks - lips_landmarks.mean(axis=0)

    preds = (learner( torch.unsqueeze(torch.tensor(normalized_lips_landmarks.flatten()), 0) ))[0]
    ret = {name: value.item() for name, value in zip(lip_columns, preds)}
    ret['image_name'] = imagepath.name
    check_limits(ret)
    return ret

def check_limits(blendshape_dict: dict):
    for key in blendshape_dict.keys():
        if key == 'image_name': continue
        blendshape_dict[key] = max(blendshape_dict[key], -100)
        blendshape_dict[key] = min(blendshape_dict[key], 100)


def get_df_for_images_in_path(path: Path):
    learner_path = Path('./Trained_models/Lips/male/torch_in=norm_fa-verts-lips_out_=lips-blends.pkl')
    learner = torch.load(learner_path, map_location='cpu')

    tracker = FaceAlignmentLandmarksTracker(device='cpu')

    blendshape_dicts_list = [get_blendshape_dict_from_learner(imagepath, learner, tracker) for imagepath in get_image_files(path, recurse=False)]
    df = pd.DataFrame(blendshape_dicts_list,
                      columns=['image_name'] + lip_columns)
    csv_path = Path('./lips.csv')
    df.to_csv(csv_path)
    return df


if __name__ == '__main__':
    df = get_df_for_images_in_path(Path('/home/sharathchandra/raw_image'))