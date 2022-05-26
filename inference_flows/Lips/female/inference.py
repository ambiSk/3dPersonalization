import sys
from pathlib import Path
sys.path.append('../../..')
import cv2
import pandas as pd
from tqdm import tqdm
from fastai.vision.all import *

lip_columns = ['L_Lip_Con_Out',
    'L_Lip_Con_Up',
    'L_Lip_Con_Fr',
    'M_Lip_Up',
    'M_Lip_Sd_L',
    'M_Lip_Fr',
    'Lip_L',
    'Lip_Up',
    'Lip_Fr',
    'Up_Lip_Thick',
    'Low_Lip_Thick']

learner_path = Path('./Trained_models/Lips/female/resnet18_female_lips_loss_1_1.pkl')

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

def mae_correct_winner(pred, label):
    winner, loser, round_num = label
    return nn.L1Loss()(pred, loser)

def r2score(pred, label):
    winner, loser, correct_winner = label
    return R2Score()(pred, correct_winner)

def init_learner(learner_path: Path, list_imagepaths: list, n_outs=15):
    learner_path = Path(learner_path)
    assert learner_path.exists()
    learner = load_learner(learner_path)
    
    def get_x(elem):
        return Path(elem)

    def get_y(row):
        return [0.0 for _ in range(n_outs)]
    
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


def get_blendshape_dict_from_learner(imagepath: Path, learner):
    preds = learner.predict(imagepath)[0]
    
    def unnormalize_learner_preds(preds):
        min_ = np.array([-115.86728 ,  -81.084114, -125.19153 ,  -88.613275,  -84.605492,\
                         -120.77778 ,  -34.514128,  -91.218717,  -96.984916, -176.76409 ,\
                         -82.516753])
        max_ = np.array([ 74.369574, 131.72103 ,  83.41437 , 114.93967 ,  53.883834,\
                          115.19734 ,  55.590147,  67.604437, 107.356   ,  99.97376 ,\
                          79.097603])
        return preds * (max_ - min_) + min_
    
    unnormalized_preds = unnormalize_learner_preds(preds)
    ret = {name: value for name, value in zip(lip_columns, unnormalized_preds)}
    ret['R_Lip_Con_Out'] = ret['L_Lip_Con_Out']
    ret['R_Lip_Con_Up'] = ret['L_Lip_Con_Up']
    ret['R_Lip_Con_Bk'] = -1 * ret['L_Lip_Con_Fr']
    
    ret['image_name'] = imagepath.name
    return ret


def get_df_for_images_in_path(path: Path):
    learner = init_learner(Path(learner_path), list(get_image_files(path, recurse=False)), 11)
    blendshape_dicts_list = [get_blendshape_dict_from_learner(imagepath, learner) for imagepath in get_image_files(path, recurse=False)]
    df = pd.DataFrame(blendshape_dicts_list,
                      columns=['image_name'] + lip_columns + ['R_Lip_Con_Out', 'R_Lip_Con_Up', 'R_Lip_Con_Bk'])
    csv_path = Path('./lips.csv')
    df.to_csv(csv_path)
    return df


if __name__ == '__main__':
    df = get_df_for_images_in_path(Path('/home/sharathchandra/raw_image'))
    
    