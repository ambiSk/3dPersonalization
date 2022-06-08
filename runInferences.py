import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import pandas as pd
import time
import shutil
import argparse
from pathlib import Path
from functools import reduce
import testing as ml_output
import parse_json as parse_json
from inference_flows.Lips.male.inference import *
import inference_flows.FaceDepthFeatures.cheeklines_v2 as cheek_lines
import inference_flows.FaceDepthFeatures.cheek_chubbiness_v3 as cheek_chubbiness
import inference_flows.FaceDepthFeatures.cheek_dimples_v1 as cheek_dimples
import inference_flows.FaceShape.FaceShape_supervisedmodel_inference as face_shape
import inference_flows.FaceShape.FaceShape_supervisedmodel_inference_female as face_shape_female
import inference_flows.Lips.male.inference as malelipsinference
import inference_flows.Lips.female.inference as femalelipsinference
import inference_flows.Nose.nose_mp_male_res2 as malenoseinference
import inference_flows.Nose.nose_mp_female_res2 as femalenoseinference
import inference_flows.Eyes.MaleEyeInfer as MaleEyesinference
import inference_flows.Eyes.FemaleEyeInfer as FemaleEyesinference
import inference_flows.Lips_v2.male.inference as malelipsinference_v2
import inference_flows.Lips_v2.female.inference as femalelipsinference_v2
import inference_flows.Lips_v3.male.inference as malelipsinference_v3
import inference_flows.Lips_v3.female.inference as femalelipsinference_v3
import inference_flows.Outfit.segmentation as outfit_segmentation
import inference_flows.Outfit.color_extractor_v2 as color_extractor_v2
from inference_flows import Hikemoji2D


# arguments = argparse.ArgumentParser()
# arguments.add_argument('--gender', type=str, default='male')
# arguments.add_argument('--input_image', type=str)
# arguments.add_argument('--inference_2d', type=str, default="no")
# args = arguments.parse_args()
# gender = args.gender
# inference_2d = args.inference_2d
# input_image = args.input_image


def read_data(path):
    files = os.listdir(path)
    files = list( filter( lambda f:f.endswith('.csv'), files) )
    df_list = []
    for file in files:
        df = pd.read_csv(os.path.join(path,file))
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df_list.append(df)
    final_df = reduce(lambda x, y: pd.merge(x, y, on = 'image_name'), df_list)
    # final_df = final_df.loc[final_df['gender'] == gender]
    #final_df["L_Cheek_Out"] = final_df[["L_Cheek_Out_x", "L_Cheek_Out_y"]].max(axis=1)
    #final_df["R_Cheek_Out"] = final_df[["R_Cheek_Out_x", "R_Cheek_Out_y"]].max(axis=1)
    #final_df = final_df.drop(['L_Cheek_Out_x', 'L_Cheek_Out_y','R_Cheek_Out_x','R_Cheek_Out_y'], axis = 1)

    final_df['R_Cheek_Out'] = final_df['R_Cheek_Out_S'] + 0.65*final_df['R_Cheek_Out'] 
    final_df['R_Cheek_Up'] = final_df['R_Cheek_Up_S'] + 0.65*final_df['R_Cheek_Up']

    final_df['L_Cheek_Out'] = final_df['L_Cheek_Out_S'] + 0.65*final_df['L_Cheek_Out'] 
    final_df['L_Cheek_Up'] = final_df['L_Cheek_Up_S'] + 0.65*final_df['L_Cheek_Up']

    final_df = final_df.drop(['R_Cheek_Out_S', 'R_Cheek_Up_S', 'L_Cheek_Out_S', 'L_Cheek_Up_S'], axis = 1)

    return final_df



def run_inferences(input_image, inference_2d, gender):
    if inference_2d == 'yes':
        
        if gender == 'male':
            component_values = Hikemoji2D.run_inference(input_image, 'male')
        else:
            component_values = Hikemoji2D.run_inference(input_image, 'female')

    if gender == 'male':
        cc_v3 = cheek_chubbiness.run_inference(input_image ,'m')
        #lips_df = malelipsinference.get_df_for_images_in_path(Path(input_images_path))
        #Eyes_df = MaleEyesinference.MaleEyeInfer(input_images_path)
        nose_params = malenoseinference.get_blend_params(input_image)
        faceShape = face_shape.main(input_image,'./Trained_models/faceshape_trial_male_v2_2_1_xg_faceangles.pkl')
        lipsV2 = malelipsinference_v2.run_inference(input_image, 'm')
        lipsV3 = malelipsinference_v3.run_inference(input_image, 'm')
    else:
        cc_v3 = cheek_chubbiness.run_inference(input_image, 'f')
        #lips_df = femalelipsinference.get_df_for_images_in_path(Path(input_images_path))
        #Eyes_df = FemaleEyesinference.FemaleEyeInfer(input_images_path)
        nose_params = femalenoseinference.get_nose_params(input_image, component_values)
        faceShape = face_shape_female.main(input_image,'./Trained_models/faceshape_trial_female_v1.3_xg_faceangles.pkl')
        lipsV2 = femalelipsinference_v2.run_inference(input_image, 'f')
        lipsV3 = femalelipsinference_v3.run_inference(input_image, 'f')



    cheeklines = cheek_lines.run_inference(input_image)

    dimples = cheek_dimples.run_inference(input_image)
    
    #Eyes_df.to_csv(os.path.join(inference_results_path,'Eyes_results22nd.csv'))
    
    # nose_params.to_csv(os.path.join(inference_results_path,'male_nose_results.csv'))
    
    # face_shape_df.to_csv(os.path.join(inference_results_path,'face_shape_results.csv'))

    #lips_df.to_csv(os.path.join(inference_results_path,'lips_results.csv'))
    dominant_color = outfit_segmentation.run_inference(input_image)

    lip_color = color_extractor_v2.run_inference(input_image)

    out3d = {**cc_v3, **nose_params, **nose_params, **faceShape, **lipsV2, **lipsV3, **cheeklines, **dimples, **dominant_color, **lip_color}
    

    out3d['R_Cheek_Out'] = out3d['R_Cheek_Out_S'] + 0.65*out3d['R_Cheek_Out'] 
    out3d['R_Cheek_Up'] = out3d['R_Cheek_Up_S'] + 0.65*out3d['R_Cheek_Up']

    out3d['L_Cheek_Out'] = out3d['L_Cheek_Out_S'] + 0.65*out3d['L_Cheek_Out'] 
    out3d['L_Cheek_Up'] = out3d['L_Cheek_Up_S'] + 0.65*out3d['L_Cheek_Up']

    del out3d['R_Cheek_Out_S']
    del out3d['R_Cheek_Up_S']
    del out3d['L_Cheek_Out_S']
    del out3d['L_Cheek_Up_S']

    if inference_2d == 'yes':
        return component_values, out3d

    return None, out3d



def get_2d_output():
    ml_output.main(input_images_path, gender)
    parse_json.get_preset_hair_color(gender)



if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--gender', type=str, default='male')
    arguments.add_argument('--input_image', type=str)
    arguments.add_argument('--inference_2d', type=str, default="no")
    
    args = arguments.parse_args()
    gender = args.gender
    inference_2d = args.inference_2d
    input_image = args.input_image
    inference_results_path = "./input_files/" + gender

    #get_2d_output()
    _, out3d = run_inferences(input_image, inference_2d, gender)
    # final_df = read_data(inference_results_path)
    json.dumps(out3d)



