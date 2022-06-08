#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import json
import cv2
import pandas as pd
import time
from PIL import Image, ImageDraw
from imutils import face_utils
# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import sys
import matplotlib.pyplot as plt
import mediapipe as mp


# In[2]:


mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        min_detection_confidence=0.5)


# In[3]:


components = pd.read_csv('./inference_flows/Nose/compresults_female.csv')
with open('./inference_flows/Nose/female_external_presets.presets_from_name.json') as f:
    presets_data = json.load(f)
nose_keys=[]
nose_blendshapes={}
for key in presets_data.keys():
    if 'nose' in key.lower():
        nose_keys.append(key)
        nose_blendshapes[key] = presets_data[key] 
        for el in nose_blendshapes[key]:
            el.remove('Facial_BS')
        nose_blendshapes[key] = dict(nose_blendshapes[key])




def get_detection_bb(detection):
    return (detection.location_data.relative_bounding_box.xmin, detection.location_data.relative_bounding_box.ymin, 
detection.location_data.relative_bounding_box.width, detection.location_data.relative_bounding_box.height)

# def face_normalize(point_ind, res_landmark, detection):
#     dbb = get_detection_bb(detection)
#     bb_tl = (dbb[0], dbb[1])
#     bb_wh = (dbb[2], dbb[3])
#     pt = (res_landmark[point_ind].x, res_landmark[point_ind].y)
#     pt_fn = ((pt[0]-bb_tl[0]) / bb_wh[1] , (pt[1]-bb_tl[1]) / bb_wh[1], res_landmark[point_ind].z)
#     return pt_fn



def face_normalize(point_ind, res_landmark, detection):
    dbb = get_detection_bb(detection)
    bb_tl = (dbb[0], dbb[1])
    bb_wh = (dbb[2], dbb[3])
    pt = (res_landmark[point_ind].x, res_landmark[point_ind].y, -res_landmark[point_ind].z)
    z_coords = [-lmpt.z for lmpt in res_landmark]
    mxz, mnz = max(z_coords)+1e-6, min(z_coords)+1e-6
    dpth = mxz - mnz
    pt_fn = ((pt[0]-bb_tl[0]) / bb_wh[1] , (pt[1]-bb_tl[1]) / bb_wh[1], (pt[2] - mnz) / bb_wh[1])    
    return pt_fn



def get_points(res_landmark, point_ind, im_shape):
    return (res_landmark[point_ind].x*im_shape[1], res_landmark[point_ind].y*im_shape[0])

def get_point3(ptinds, res):
    return [(res.landmark[ptind].x, res.landmark[ptind].y, res.landmark[ptind].z) for ptind in ptinds]

def get_point3_fn(ptinds, res, detection):
    return [face_normalize(ptind, res.landmark, detection) for ptind in ptinds]

def proj_u_over_plane(u, n):
    """ Project vector u on (Plane P, represented by a vector orthogonal to it, n) """
    assert len(u)== len(n), 'both vector and plane should be same length vector'
    u = np.array(u)
    n = np.array(n)
    # find the norm of the vector n  
    n_norm = np.linalg.norm(n)
    proj_of_u_on_n = (np.dot(u, n) / n_norm**2) * n 
    # subtract proj_of_u_on_n from u:  
    # this is the projection of u on Plane P 
    proj = u - proj_of_u_on_n
    return proj

def dist_point_line(p1, p2, p3):
    '''
    ## p1 and p2 giving a line and p3 the third point, fn finds perpendicular distance 
    '''
    return  np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

def get_dist_2pts(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def get_normal_vec_plane(p1, p2, p3):
    """Get the normal vector to the plane given by three point vectors, p1 p2 p3 on the plane"""
    # These two vectors are in the plane
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    cp = cp / np.linalg.norm(cp)
    a, b, c = cp
    return a,b,c

def get_cosine_two_vecs(vector_1, vector_2):    
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def get_nose_ht(res, detection, pos = 'mid'):
    """nose height at mid and the top, input pos = bot, mid or top"""
    if pos == 'mid':
        pt_ind = 195
    elif pos == 'bot':
        pt_ind = 6
    else:
        pt_ind = 4
    ht_pts = get_point3_fn([pt_ind], res, detection) # 4, 197, 168
    n_pl_pts = get_point3_fn([123, 352, 152 ], res, detection)
    plane_vec = get_normal_vec_plane(n_pl_pts[0], n_pl_pts[1], n_pl_pts[2])
    proj_hts = [proj_u_over_plane(htpt, plane_vec) for htpt in ht_pts]
#     mean_nose_ht = np.mean([get_dist_2pts(proj_hts[ii], ht_pts[ii]) for ii in range(len(ht_pts))])
    mean_nose_ht = np.mean([get_dist_2pts( proj_hts[ii] , ht_pts[ii]) for ii in range(len(ht_pts))])
    return  mean_nose_ht

def get_pt_circle(p1):
    return (p1[0]-1, p1[1]-1, p1[0]+1, p1[1]+1)

def draw_pt_image(image, pt, fill='blue'):
    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    if fill == 'blue':
        fill = (0,0,255)
    else:
        fill = (255,255,0)
    draw.ellipse(get_pt_circle(pt) , fill = fill, outline =fill)
    return im

def draw_ptind_image(image, ptinds, fill='blue'):
    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    if fill == 'blue':
        fill = (0,0,255)
    else:
        fill = (10,10,10)
    for p in ptinds:
        pt = get_points(res.landmark, p, image.shape)
        draw.ellipse(get_pt_circle(pt) , fill = fill, outline =fill)
    return im

def get_mp_res(image):
    results_fd = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection = results_fd.detections[0]
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    res = results.multi_face_landmarks[0]
    return res, detection




def get_blend_params_image(image):

    res, detection = get_mp_res(image)
    nose_bot_corner_pts = get_point3_fn([48, 331], res, detection)
    face_width_pts = get_point3_fn([ 123, 352], res, detection)
    nose_bot_len = get_dist_2pts(nose_bot_corner_pts[0], nose_bot_corner_pts[1])
    face_width =  get_dist_2pts(face_width_pts[0], face_width_pts[1])
    nose_w = nose_bot_len / face_width

    
#     low_pts = get_point3_fn([48,331,1], res, detection)
#     v1, v2, v3 = np.array(low_pts[0]), np.array(low_pts[1]), np.array(low_pts[2])
#     pointy_angle = get_cosine_two_vecs(v1-v3, v2-v3)
    bot_mid = (np.array(nose_bot_corner_pts[0])+ np.array(nose_bot_corner_pts[1]))[:2]/2.
    low_pt = get_point3_fn([1], res, detection)
#     low_pt = np.array(low_pt[0][:2])
#     pointy_angle = get_dist_2pts(bot_mid, low_pt)
    pointy_angle = dist_point_line(np.array(nose_bot_corner_pts[0]), np.array(nose_bot_corner_pts[1]), np.array(low_pt))
    pointy_angle1 = dist_point_line(np.array(nose_bot_corner_pts[0]), np.array(nose_bot_corner_pts[1]), np.array(get_point3_fn([94], res, detection)))    

    bot_pts = [(114, 343), (189, 413), (128, 357)]
    bot_pts_fn = [get_point3_fn(pt, res, detection) for pt in bot_pts]
    bot_pts_dists = [get_dist_2pts(pt[0], pt[1]) for pt in bot_pts_fn]
    bot_width_mean = np.mean(bot_pts_dists)


    bridge_pts = [(196, 419), (3, 248), (122,351)]
    bridge_pts_lo = [[ 3, 248 ], [ 51, 281], [ 45, 275] ]
    bridge_pts_up=[[ 193, 417 ], [ 122, 351 ], [ 196, 419 ]]        
#     bridge_pts = bridge_pts_up
    bridge_pts_fn = [get_point3_fn(pt, res, detection) for pt in bridge_pts]
    bridge_pts_dists = [get_dist_2pts(pt[0], pt[1]) for pt in bridge_pts_fn]
    bridge_width_mean = np.mean(bridge_pts_dists) / face_width

    bridge_pts = [(196, 419), (3, 248), (122,351)]
    
    bridge_pts_lo = [[ 3, 248 ], [ 51, 281], [ 45, 275] ]
    bridge_pts_up=[[ 193, 417 ], [ 122, 351 ], [ 196, 419 ]]    
    bridge_pts = bridge_pts_lo
    
    bridge_pts_fn = [get_point3_fn(pt, res, detection) for pt in bridge_pts]
    bridge_pts_dists = [get_dist_2pts(pt[0], pt[1]) for pt in bridge_pts_fn]
    bridge_width_mean = np.mean(bridge_pts_dists)
    
    bridge_pts_up_fn = [get_point3_fn(pt, res, detection) for pt in bridge_pts_up]
    bridge_pts_up_dists = [get_dist_2pts(pt[0], pt[1]) for pt in bridge_pts_up_fn]
    bridge_width_up_mean = np.mean(bridge_pts_up_dists)    
    
# 196, 419, 3, 248, 122,351
# 114, 343, 189, 413, 128, 357
# 2, 8
#     print(get_point3([48, 331], res=res))
#     print(image.shape)
#     plt.imshow(image)

    bot_ht, mid_ht, top_ht = get_nose_ht(res, detection, 'bot'), get_nose_ht(res, detection, 'mid'), get_nose_ht(res, detection, 'top')
    brg_top_ratio = mid_ht / top_ht

    Nose_Bridge_Up_Dn = 82.6
    Nose_Bridge_Fr_Bk = 31.9
    M_Nose_Fr = 26.3
    nose_con_out_max, nose_con_out_min = 150, -150
#         R_Nose_Con_Out = -20 + (57+20) * (nose_bot_len-0.22) / (0.32 - 0.22)
    nosew_param = 1.* nose_w + .01* nose_bot_len
    R_Nose_Con_Out = nose_con_out_min + (nose_con_out_max - nose_con_out_min) * (nosew_param - 0.29) / (0.35 - 0.29)

    L_Nose_Con_Out = R_Nose_Con_Out
    nose_con_up_max, nose_con_up_min = R_Nose_Con_Out * 70 / nose_con_out_max, 0
    L_Nose_Con_Up = nose_con_up_min + (nose_con_up_max - nose_con_up_min)*(pointy_angle1 - 0.06)/(0.08-0.06)
    
    R_Nose_Con_Up = L_Nose_Con_Up
#     Nose_Bridge_Side_Scale = 16.3 + (73 - 16.3) * (bot_width_mean - 0.14) / (0.23 - 0.14)
    bridge_max, bridge_min = 0, -120 # 47., 6.
    Nose_Bridge_Side_Scale = bridge_min + (bridge_max - bridge_min) * (bridge_width_mean - 0.055) / (0.095 - 0.055)
    Nose_Bridge_Up_Dn = Nose_Bridge_Side_Scale + 0
    Nose_Bridge_Fr_Bk =  -Nose_Bridge_Up_Dn # Nose_Bridge_Side_Scale - 90

    Nose_Bridge_Fr_Bk = 20 + (-80 - 20) * (bot_ht - 0.4) / (0.6 - 0.4) + Nose_Bridge_Fr_Bk  # - Nose_Bridge_Side_Scale * 0.2  
    Nose_Bridge_Up_Dn = -20 + (80 - (-20)) * (brg_top_ratio - 0.94) / (0.96 - 0.94) + Nose_Bridge_Up_Dn
#     Nose_Bridge_Up_Dn = Nose_Bridge_Up_Dn + Nose_Bridge_Fr_Bk * -1

#     if Nose_Bridge_Side_Scale < (bridge_max + bridge_min) / 3.:
#         Nose_Bridge_Side_Scale = max(Nose_Bridge_Side_Scale - 20, 0)
#     Nose_Bridge_Up_Dn = 111
    
    nose_len_pts = [2, 8]
    lip_top_pt = [0]
    nose_len_pts_fn = get_point3_fn(nose_len_pts, res, detection)
    lip_pt_fn = get_point3_fn(lip_top_pt, res, detection)
    nose_len = get_dist_2pts(nose_len_pts_fn[0], nose_len_pts_fn[1])
    nose_lip_len = get_dist_2pts(lip_pt_fn[0], nose_len_pts_fn[1])
    nose_len = nose_len / nose_lip_len

    M_Nose_Up = -1 + (-120 + 1)*(nose_len-0.78)/(0.85 - 0.78)

    nostril_pts = [(458, 328) , (438, 460)]
    nostril_pts_fn = [get_point3_fn(nostrl_pt, res, detection) for nostrl_pt in nostril_pts]
    nostril_wds = [get_dist_2pts(pt[0], pt[1])  for pt in nostril_pts_fn]
    nostril_param = np.mean(nostril_wds)
    
    Nostril_Curvature = -70 + (50 - (-70))*(nostril_param - 0.075) / (0.095 - 0.075)
    
    Nosebridge_Bump_In_Out = -20 + (140 - (-20)) * (bridge_width_up_mean - 0.07) / (0.105 - 0.07)
        
    Nose_Tip_Fr_Bk = 10
    Nosebridge_Bump_Up_Dn = -40
    Nose_Tip_Lt_Rt = 0
    R_Nose_Con_Bk = 0
    Nose_Tip_Up_Dn = -60
    Nosebridge_Bump_Fr_Bk = 10
    L_Nose_Con_Fr = 0
    Nose_Tip_Scale=20

    
    return Nose_Bridge_Up_Dn, Nose_Bridge_Fr_Bk, M_Nose_Fr, R_Nose_Con_Out, L_Nose_Con_Out, L_Nose_Con_Up, R_Nose_Con_Up, Nose_Bridge_Side_Scale, M_Nose_Up, Nostril_Curvature, Nosebridge_Bump_In_Out, Nose_Tip_Fr_Bk, Nosebridge_Bump_Up_Dn, Nose_Tip_Lt_Rt, R_Nose_Con_Bk, Nose_Tip_Up_Dn, Nosebridge_Bump_Fr_Bk, L_Nose_Con_Fr, Nose_Tip_Scale






def get_nose_params(data_dir, preset_mix = 0.4):

    img_files = [data_dir +'/'+filenm for filenm in os.listdir(data_dir)]

    out_data = []
    
    res_dict = {}

    for img_path in img_files:
#         print(img_path.split('/')[-1],)
        if not (img_path.endswith("png") or img_path.endswith("jpg") or img_path.endswith("jpeg")):
            continue
        image = cv2.imread(img_path)
        if type(image)==type(None):
            print('couldnt read image.. exiting..', img_path)
            sys.exit()
        img_blendshapes = get_blend_params_image(image)
        img_name = img_path.split('/')[-1]
        bs_names = ['Nose_Bridge_Up_Dn', 'Nose_Bridge_Fr_Bk', 'M_Nose_Fr', 'R_Nose_Con_Out', 'L_Nose_Con_Out', 'L_Nose_Con_Up', 
                    'R_Nose_Con_Up', 'Nose_Bridge_Side_Scale', 'M_Nose_Up', 'Nostril_Curvature', 'Nosebridge_Bump_In_Out',
                    'Nose_Tip_Fr_Bk', 'Nosebridge_Bump_Up_Dn', 'Nose_Tip_Lt_Rt', 'R_Nose_Con_Bk', 'Nose_Tip_Up_Dn', 'Nosebridge_Bump_Fr_Bk', 'L_Nose_Con_Fr', 'Nose_Tip_Scale'
                   ]
        img_bs_dict = dict(zip(bs_names, img_blendshapes))
        img_nose_cls = components.loc[components.image_name==img_name]['Nose'].values[0]
        preset_bs = nose_blendshapes[img_nose_cls]
        img_bs_merge={}
        for key in set(preset_bs.keys()).union(set(img_bs_dict.keys())):
            if key in preset_bs and key in img_bs_dict:
                img_bs_merge[key] = preset_mix * preset_bs[key] + (1 - preset_mix) * img_bs_dict[key]
            elif key in preset_bs:
                img_bs_merge[key] = preset_bs[key]
            else:
                img_bs_merge[key] = img_bs_dict[key]
        res_dict[img_name] = img_bs_merge
#         Nose_Bridge_Up_Dn, Nose_Bridge_Fr_Bk, M_Nose_Fr, R_Nose_Con_Out, L_Nose_Con_Out, L_Nose_Con_Up, R_Nose_Con_Up, Nose_Bridge_Side_Scale, M_Nose_Up, Nostril_Curvature, Nosebridge_Bump_In_Out = get_blend_params_image(image)
#         out_data.append([img_name, Nose_Bridge_Up_Dn, Nose_Bridge_Fr_Bk, M_Nose_Fr, R_Nose_Con_Out, L_Nose_Con_Out, L_Nose_Con_Up, R_Nose_Con_Up, Nose_Bridge_Side_Scale, M_Nose_Up, Nostril_Curvature, Nosebridge_Bump_In_Out])

    blend_shapes_params = pd.DataFrame.from_dict(res_dict, orient='index').reset_index()
    blend_shapes_params = blend_shapes_params.rename(columns={'index':'image_name'})
    return blend_shapes_params




