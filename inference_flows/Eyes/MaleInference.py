from pathlib import Path
import pandas as pd
from fastai.vision.all import *


def init_learner(learner_path: Path, list_imagepaths: list):
    learner_path = Path(learner_path)
    assert learner_path.exists()
    learner = load_learner(learner_path)
    
    def get_x(elem):
        return Path(elem)

    def get_y(row):
        return [0.0]
    
    dblock = DataBlock(
        blocks=(ImageBlock, RegressionBlock),
        get_x = get_x,
        get_y = get_y,
        splitter=RandomSplitter(),
        item_tfms=(Resize(224)),
    )
    dls = dblock.dataloaders(list_imagepaths)
    learner.dls = dls
    
    return learner

def get_eye_type(imagepath: Path, learner):
    ret = {}
    preds = learner.predict(imagepath)[0]
    ret['EyeType'] = preds
    ret['image_name'] = imagepath.name
    return ret
    #takeargmax and specify the Eye class

def get_df_for_images_in_path(path: Path):
    #landmarks_tracker = FaceAlignmentLandmarksTracker()
    learner = init_learner(Path(learner_path), list(get_image_files(path, recurse=False)))
    blendshape_dicts_list = [get_eye_type(imagepath, learner) for imagepath in get_image_files(path, recurse=False)]
    df = pd.DataFrame(blendshape_dicts_list,
                      columns=['image_name', 'EyeType'])

    csv_path = Path('./MaleEyes.csv')
    
    return df

learner_path = Path('Trained_models/Eyes/Male/resnet18_male_eyes.pkl')

if __name__ == '__main__':
    df = get_df_for_images_in_path(Path('/home/sharathchandra/raw_image'))
