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


arguments = argparse.ArgumentParser()
arguments.add_argument('--gender', type=str, default='male')
arguments.add_argument('--input_dir', type=str, default="./input_images/")
arguments.add_argument('--inference_2d', type=str, default="no")
args = arguments.parse_args()
gender = args.gender
inference_2d = args.inference_2d
input_images_path = args.input_dir
inference_results_path = "./input_files/" + gender


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



def run_inferences():
    if inference_2d == 'yes':
        filepath = os.path.dirname(os.path.abspath(__file__))
        if gender == 'male':
            Hikemoji2D.run_inference(input_images_path, os.path.join(filepath, "component_values_male.csv"), 'male')
            shutil.copy(os.path.join(filepath, "component_values_male.csv"),
                        os.path.join(filepath, "inference_flows/Nose/compresults_male.csv"))
        else:
            Hikemoji2D.run_inference(input_images_path, os.path.join(filepath, "component_values_female.csv"), 'male')
            shutil.copy(os.path.join(filepath, "component_values_female.csv"),
                        os.path.join(filepath, "inference_flows/Nose/compresults_female.csv"))

    if gender == 'male':
        cheek_chubbiness.run_inference(input_images_path,os.path.join(inference_results_path,'cheek_chubbiness_results.csv'),'m')
        #lips_df = malelipsinference.get_df_for_images_in_path(Path(input_images_path))
        #Eyes_df = MaleEyesinference.MaleEyeInfer(input_images_path)
        nose_params = malenoseinference.get_blend_params(input_images_path)
        face_shape_df = face_shape.main(input_images_path,'/home/sharathchandra/faceshape_trial_male_v2_2_1_xg_faceangles.pkl')
        malelipsinference_v2.run_inference(input_images_path, os.path.join(inference_results_path, 'lips_personalisation_v2_results.csv'), 'm')
        malelipsinference_v3.run_inference(input_images_path, os.path.join(inference_results_path, 'lips_personalisation_v3_results.csv'), 'm')
    else:
        cheek_chubbiness.run_inference(input_images_path,os.path.join(inference_results_path,'cheek_chubbiness_results.csv'),'f')
        #lips_df = femalelipsinference.get_df_for_images_in_path(Path(input_images_path))
        #Eyes_df = FemaleEyesinference.FemaleEyeInfer(input_images_path)
        nose_params = femalenoseinference.get_nose_params(input_images_path)
        face_shape_df = face_shape_female.main(input_images_path,'/home/sharathchandra/faceshape_trial_female_v1.3_xg_faceangles.pkl')
        femalelipsinference_v2.run_inference(input_images_path, os.path.join(inference_results_path, 'lips_personalisation_v2_results.csv'), 'f')
        femalelipsinference_v3.run_inference(input_images_path, os.path.join(inference_results_path, 'lips_personalisation_v3_results.csv'), 'f')



    cheek_lines.run_inference(input_images_path,os.path.join(inference_results_path,'cheek_lines_results.csv'))

    cheek_dimples.run_inference(input_images_path,os.path.join(inference_results_path,'cheek_dimples_results.csv'))
    
    #Eyes_df.to_csv(os.path.join(inference_results_path,'Eyes_results22nd.csv'))
    
    nose_params.to_csv(os.path.join(inference_results_path,'male_nose_results.csv'))
    
    face_shape_df.to_csv(os.path.join(inference_results_path,'face_shape_results.csv'))

    #lips_df.to_csv(os.path.join(inference_results_path,'lips_results.csv'))
    outfit_segmentation.run_inference(input_images_path,os.path.join(inference_results_path,'outfit_color_results.csv'))
    color_extractor_v2.run_inference(input_images_path,os.path.join(inference_results_path,'lip_color_results.csv'))



def get_2d_output():
    ml_output.main(input_images_path, gender)
    parse_json.get_preset_hair_color(gender)



if __name__ == "__main__":
    #get_2d_output()
    run_inferences()
    final_df = read_data(inference_results_path)
    final_df.to_csv('results.csv')



