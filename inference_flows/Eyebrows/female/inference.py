import sys
from pathlib import Path
sys.path.append('../../..')

import cv2
import pandas as pd
from landmark_utils.fa_tracker import FaceAlignmentLandmarksTracker
from tqdm import tqdm
from fastai.vision.all import *

learner_path = Path('./inference_flows/Eyebrows/female/export.pkl')
num_tasks = 13

def get_x(row):
    return Path(row['human_img_path'])

def get_y(row):
    return get_lips_params_avatar_imagename(row['winner']), \
           get_lips_params_avatar_imagename(row['loser']), \
           row['round_num']

def group_loss(pred, label):
    winner, loser, round_num = label
    return nn.MSELoss()(pred, winner)

def loss(pred, label, trp_wt=1, group_loss_wt=1, point_loss_wt=0, MARGIN_TRIPLET=0.2):

    def triplet_loss(pred, label, MARGIN=MARGIN_TRIPLET):
        ''' Analysis can be seen in the notebook - loss_function '''
        winner, loser, round_num = label

        winner_dist = ((pred - winner) ** 2).mean(axis=1)
        loser_dist = ((pred - loser) ** 2).mean(axis=1)

        loss = winner_dist - loser_dist + MARGIN
        return torch.relu(loss).mean()

    def group_loss(pred, label):
        winner, loser, round_num = label
        return (nn.MSELoss()(pred, winner) + nn.MSELoss()(pred, loser)) / 2

    def point_loss(pred, label, thresh = 2.9):
        winner, loser, round_num = label
        bs = winner.shape[0]

        num_rounds_thresh = torch.where(round_num >= thresh, 1, 0).sum()
        pt_loss_sum = torch.where(round_num > thresh, 
                          ((pred - winner) ** 2).mean(axis=1), 
                          torch.zeros((bs, )).float().to(device)).sum()

        return torch.where( num_rounds_thresh > 0, pt_loss_sum / num_rounds_thresh, torch.tensor(0).float().to(device) )
    
    return trp_wt * triplet_loss(pred, label) + \
           group_loss_wt * group_loss(pred, label) + \
           point_loss_wt * point_loss(pred, label)


def splitter(df):
    ''' We need to ensure that different images be used for training and validation, and not merely different
        triplets for same images '''
    TRAIN_SIZE = 0.8
    VALID_SIZE = 1.0 - TRAIN_SIZE
    train_idx = list( range(int(TRAIN_SIZE * len(df))) )
    valid_idx = list( range(int(TRAIN_SIZE * len(df)), len(df)) )
    return train_idx, valid_idx


def mae_winner(pred, label):
    winner, loser, round_num = label
    return nn.L1Loss()(pred, winner)


def mae_loser(pred, label):
    winner, loser, round_num = label
    return nn.L1Loss()(pred, loser)


def init_learner(learner_path: Path, list_imagepaths: list):
    learner_path = Path(learner_path)
    assert learner_path.exists()
    learner = load_learner(learner_path)
    
    def get_x(elem):
        return Path(elem)

    def get_y(row):
        return [0.0 for _ in range(num_tasks)]
    
    dblock = DataBlock(
        blocks=(ImageBlock, RegressionBlock),
        get_x = get_x,
        get_y = get_y,
        splitter=RandomSplitter(),
        item_tfms=(Resize(224)),
        batch_tfms=[
            FlipItem(p=0.5),
            Brightness(max_lighting=0.3, p=0.7, draw=None, batch=False),
            Saturation(max_lighting=0.3, p=0.7, draw=None, batch=False),
            Hue(max_hue=0.1, p=0.75, draw=None, batch=False),
            RandomErasing(p=0.2, sl=0.0, sh=0.15, max_count=6, min_aspect=0.2)
        ],
    )
    dls = dblock.dataloaders(list_imagepaths)
    learner.dls = dls
    
    return learner


def get_image_from_path(path: Path):
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_upper_lip_blendshape_value(upper_lip_rel_thickness):
    if upper_lip_rel_thickness < 0.10:
        return -20
    elif upper_lip_rel_thickness < 0.11:
        return ((-13 - (-20)) * (upper_lip_rel_thickness - 0.10) / (0.11 - 0.10)) + (-20)
    elif upper_lip_rel_thickness < 0.12:
        return ((-5 - (-13)) * (upper_lip_rel_thickness - 0.11) / (0.12 - 0.11)) + (-13)
    elif upper_lip_rel_thickness < 0.13:
        return ((5 - (-5)) * (upper_lip_rel_thickness - 0.12) / (0.13 - 0.12)) + (-5)
    elif upper_lip_rel_thickness < 0.14:
        return ((15 - (5)) * (upper_lip_rel_thickness - 0.13) / (0.14 - 0.13)) + (5)
    elif upper_lip_rel_thickness < 0.15:
        return ((20 - (15)) * (upper_lip_rel_thickness - 0.14) / (0.14 - 0.13)) + (15)
    elif upper_lip_rel_thickness < 0.16:
        return ((30 - (20)) * (upper_lip_rel_thickness - 0.15) / (0.15 - 0.14)) + (20)
    else:
        return 30


def get_lower_lip_blendshape_value(lower_lip_rel_thickness):
    if lower_lip_rel_thickness < 0.10:
        return -40
    elif lower_lip_rel_thickness < 0.11:
        return ((-10 - (-40)) * (lower_lip_rel_thickness - 0.10) / (0.11 - 0.10)) + (-40)
    elif lower_lip_rel_thickness < 0.12:
        return ((5 - (-10)) * (lower_lip_rel_thickness - 0.11) / (0.12 - 0.11)) + (-10)
    elif lower_lip_rel_thickness < 0.13:
        return ((15 - 5) * (lower_lip_rel_thickness - 0.12) / (0.13 - 0.12)) + 5
    elif lower_lip_rel_thickness < 0.14:
        return ((20 - 15) * (lower_lip_rel_thickness - 0.13) / (0.14 - 0.13)) + 15
    elif lower_lip_rel_thickness < 0.15:
        return ((25 - 20) * (lower_lip_rel_thickness - 0.14) / (0.15 - 0.14)) + 20
    elif lower_lip_rel_thickness < 0.16:
        return ((30 - 25) * (lower_lip_rel_thickness - 0.15) / (0.16 - 0.15)) + 25
    elif lower_lip_rel_thickness < 0.17:
        return ((40 - 30) * (lower_lip_rel_thickness - 0.16) / (0.17 - 0.16)) + 30
    else:
        return 40

def get_lip_h_blendshape_value(lip_h_ratio):
    if lip_h_ratio < 0.50:
        return -20
    elif lip_h_ratio < 0.58:
        return ((-15 - (-20)) * (lip_h_ratio - 0.50) / (0.58 - 0.50)) + (-20)
    elif lip_h_ratio < 0.63:
        return ((-13 - (-15)) * (lip_h_ratio - 0.58) / (0.63 - 0.58)) + (-15)
    elif lip_h_ratio < 0.70:
        return ((0 - (-13)) * (lip_h_ratio - 0.63) / (0.70 - 0.63)) + (-13)
    elif lip_h_ratio < 0.85:
        return ((10 - (0)) * (lip_h_ratio - 0.70) / (0.85 - 0.70)) + (0)

    
def get_blendshape_dict_for_image(imagepath: Path, landmarks_tracker):
    image = get_image_from_path(imagepath)
    landmarks_tracker.set_image(image)
    try:
        upper_lip, lower_lip = landmarks_tracker.get_normalized_lip_thickness()
        left_lip_h = landmarks_tracker.get_left_lip_h_ratio()
        right_lip_h = landmarks_tracker.get_right_lip_h_ratio()
        mean_lip_h = (left_lip_h + right_lip_h) / 2
        up_thick_blendshape = get_upper_lip_blendshape_value(upper_lip)
        low_thick_blendshape = get_lower_lip_blendshape_value(lower_lip)
        width_blendshape = get_lip_h_blendshape_value(mean_lip_h)

    except Exception as e:
        print(f'Exception occured during thickness evaluation: \n{e}')
        up_thick_blendshape = 0
        low_thick_blendshape = 0
        width_blendshape = 0

    blendshape_dict = {'L_Lip_Con_Out': width_blendshape, 'R_Lip_Con_Out': width_blendshape,
                       'Up_Lip_Thick': up_thick_blendshape, 'Low_Lip_Thick': low_thick_blendshape,
                       'image_name': imagepath.name}
    return blendshape_dict


def get_blendshape_dict_from_learner(imagepath: Path, learner):
    
    preds = learner.predict(imagepath)[0]
    
    def unnormalize_learner_preds(preds):
        mean = np.array([-1.082718, -5.031452,  9.100340,-15.765190, -7.796244, 20.905657, 20.186355, -1.037952,-19.087627, -8.135814,-10.399437, 13.946333, 21.417311])
        std = np.array([10.101808, 14.759556, 12.574787, 24.021607, 14.593512, 16.269790, 36.910422, 25.159186, 25.691212, 14.864209, 15.860570, 12.323027, 52.391805])
        return np.array(preds) * std + mean
    
    unnormalized_preds = unnormalize_learner_preds(preds)
    cols = ['L_Lip_Con_Out', 'L_Lip_Con_Up', 'L_Lip_Con_Fr', 'R_Lip_Con_Out', 'R_Lip_Con_Up', 'R_Lip_Con_Bk', 'M_Lip_Up', 'M_Lip_Sd_L', 'M_Lip_Fr', 'Lip_L', 'Lip_Up', 'Lip_Fr', 'Lip_pressed_BS', 'Up_Lip_Thick', 'Low_Lip_Thick']
    ret = {name: value for name, value in zip(cols, unnormalized_preds)}
    ret['image_name'] = imagepath.name
    return ret
    

def is_image_file(filepath: Path)-> bool:
    return filepath.name.endswith('.png') or filepath.name.endswith('.jpg')


def get_df_for_images_in_path(path: Path):
    #landmarks_tracker = FaceAlignmentLandmarksTracker()
    learner = init_learner(Path(learner_path), list(get_image_files(path, recurse=False)))
    blendshape_dicts_list = [get_blendshape_dict_from_learner(imagepath, learner) for imagepath in get_image_files(path, recurse=False)]
    df = pd.DataFrame(blendshape_dicts_list,
                      columns=['image_name'] + ['Brow_Main_BS.L_Brow_Out','Brow_Main_BS.L_Brow_Up','Brow_Main_BS.L_Brow_Fr','Brow_Main_BS.L_Brow_Con_IN_Out','Brow_Main_BS.L_Brow_Con_IN_Up','Brow_Main_BS.L_Brow_Con_IN_Fr','Brow_Main_BS.L_M_Brow_Out','Brow_Main_BS.L_M_Brow_Up','Brow_Main_BS.L_M_Brow_Fr','Brow_Main_BS.L_Brow_Con_OUT_Out','Brow_Main_BS.L_Brow_Con_OUT_Up','Brow_Main_BS.L_Brow_Con_OUT_Fr','Brow_Main_BS.Eye_Brow_Think'])
    
    df['Brow_Main_BS.R_Brow_Out'] = df['Brow_Main_BS.R_Brow_Out']
    df['Brow_Main_BS.R_Brow_Up'] = df['Brow_Main_BS.L_Brow_Up']
#     df['Brow_Main_BS.L_Brow_Fr'] = df['Brow_Main_BS.R_Brow_Fr']
    df['Brow_Main_BS.R_Brow_Con_IN_Out'] = df['Brow_Main_BS.L_Brow_Con_IN_Out']
    df['Brow_Main_BS.R_Brow_Con_IN_Fr'] = df['Brow_Main_BS.L_Brow_Con_IN_Fr']
    df['Brow_Main_BS.R_M_Brow_Out'] = df['Brow_Main_BS.L_M_Brow_Out']
    df['Brow_Main_BS.R_M_Brow_Up'] = df['Brow_Main_BS.L_M_Brow_Up']
#     df['Brow_Main_BS.R_M_Brow_Fr'] = df['Brow_Main_BS.L_M_Brow_Fr']
    df['Brow_Main_BS.R_Brow_Con_OUT_Out'] = df['Brow_Main_BS.L_Brow_Con_OUT_Out']
    df['Brow_Main_BS.R_Brow_Con_OUT_Up'] = df['Brow_Main_BS.L_Brow_Con_OUT_Up']
#     df['Brow_Main_BS.R_Brow_Con_OUT_Fr'] = df['Brow_Main_BS.L_Brow_Con_OUT_Fr']

    df['Facial_BS.R_Brow_Out'] = df['Facial_BS.R_Brow_Out']
    df['Facial_BS.R_Brow_Up'] = df['Facial_BS.L_Brow_Up']
#     df['Facial_BS.L_Brow_Fr'] = df['Facial_BS.R_Brow_Fr']
    df['Facial_BS.R_Brow_Con_IN_Out'] = df['Facial_BS.L_Brow_Con_IN_Out']
    df['Facial_BS.R_Brow_Con_IN_Fr'] = df['Facial_BS.L_Brow_Con_IN_Fr']
    df['Facial_BS.R_M_Brow_Out'] = df['Facial_BS.L_M_Brow_Out']
    df['Facial_BS.R_M_Brow_Up'] = df['Facial_BS.L_M_Brow_Up']
#     df['Facial_BS.R_M_Brow_Fr'] = df['Facial_BS.L_M_Brow_Fr']
    df['Facial_BS.R_Brow_Con_OUT_Out'] = df['Facial_BS.L_Brow_Con_OUT_Out']
    df['Facial_BS.R_Brow_Con_OUT_Up'] = df['Facial_BS.L_Brow_Con_OUT_Up']
#     df['Facial_BS.R_Brow_Con_OUT_Fr'] = df['Facial_BS.L_Brow_Con_OUT_Fr']

    csv_path = Path('./lips.csv')
    df.to_csv(csv_path)
    return df

if __name__ == '__main__':
    df = get_df_for_images_in_path(Path('/home/sharathchandra/raw_image'))
    
    