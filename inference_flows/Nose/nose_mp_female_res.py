#!/usr/bin/env python
# coding: utf-8

import cv2
import pandas as pd


import time




from PIL import Image, ImageDraw




from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import sys
import matplotlib.pyplot as plt



import mediapipe as mp



mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    min_detection_confidence=0.5)





mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



data_dir = '/nfs_storage/fs-mnt6/pranshu/selfie_sticker/selfie_sticker_console/static/Hikemoji3D_female/raw_image'
# data_dir = '/nfs_storage/fs-mnt6/mujtaba/3D_vision/Nose/results/female_v0_out/'
img_files = [data_dir +'/'+filenm for filenm in os.listdir(data_dir)]




def get_detection_bb(detection):
    return (detection.location_data.relative_bounding_box.xmin, detection.location_data.relative_bounding_box.ymin, 
detection.location_data.relative_bounding_box.width, detection.location_data.relative_bounding_box.height)

def face_normalize(point_ind, res_landmark, detection):
    dbb = get_detection_bb(detection)
    bb_tl = (dbb[0], dbb[1])
    bb_wh = (dbb[2], dbb[3])
    pt = (res_landmark[point_ind].x, res_landmark[point_ind].y)
    pt_fn = ((pt[0]-bb_tl[0]) / bb_wh[0] , (pt[1]-bb_tl[1]) / bb_wh[1], res_landmark[point_ind].z)
    return pt_fn




def get_points(res_landmark, point_ind, im_shape):
    return (res_landmark[point_ind].x*im_shape[1], res_landmark[point_ind].y*im_shape[0])




def get_point3(ptinds, res):
    return [(res.landmark[ptind].x, res.landmark[ptind].y, res.landmark[ptind].z) for ptind in ptinds]




def get_point3_fn(ptinds, res, detection):
    return [face_normalize(ptind, res.landmark, detection) for ptind in ptinds]
    




def dist_point_line(p1, p2, p3):
    '''
    ## p1 and p2 giving a line and p3 the third point, fn finds perpendicular distance 
    '''
    return  np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)




def get_dist_2pts(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))




def get_normal_vec_plane(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    return a,b,c
    
p1 = np.array([1, 0, 0])
p2 = np.array([0, 1, 0])
p3 = np.array([0, 0, 0])
get_normal_vec_plane(p1, p2, p3)




def get_cosine_two_vecs(vector_1, vector_2):    
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle




def get_pt_circle(p1):
    return (p1[0]-1, p1[1]-1, p1[0]+1, p1[1]+1)




def draw_pt_image(image, pt, fill='blue'):
    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    draw.ellipse(get_pt_circle(pt) , fill = fill, outline =fill)
    return im


def draw_ptind_image(image, ptinds, fill='blue'):
    
    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    for p in ptinds:
        pt = get_points(res.landmark, p, image.shape)
        draw.ellipse(get_pt_circle(pt) , fill = fill, outline =fill)
    return im


def get_nose_params(data_dir):
    nbls = []
    pas = []
    bws = []
    out_data = []
    nws=[]
    noslens=[]
    
    img_files = [data_dir +'/'+filenm for filenm in os.listdir(data_dir)]


    for img_path in img_files:
#         print(img_path.split('/')[-1],)
        image = cv2.imread(img_path)
        if type(image)==type(None):
            print('couldnt read image.. exiting..', img_path)
            sys.exit()

        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        results_fd = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detection = results_fd.detections[0]

        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            min_detection_confidence=0.5)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        res = results.multi_face_landmarks[0]


        nose_bot_corner_pts = get_point3_fn([48, 331], res, detection)
        nose_bot_len = get_dist_2pts(nose_bot_corner_pts[0], nose_bot_corner_pts[1])
        nbls.append(nose_bot_len)

    #     low_pts = get_point3_fn([48,331,1], res, detection)
    #     v1, v2, v3 = np.array(low_pts[0]), np.array(low_pts[1]), np.array(low_pts[2])
    #     pointy_angle = get_cosine_two_vecs(v1-v3, v2-v3)
        bot_mid = (np.array(nose_bot_corner_pts[0])+ np.array(nose_bot_corner_pts[1]))[:2]/2.
        low_pt = get_point3_fn([1], res, detection)
    #     low_pt = np.array(low_pt[0][:2])
    #     pointy_angle = get_dist_2pts(bot_mid, low_pt)
        pointy_angle = dist_point_line(np.array(nose_bot_corner_pts[0]), np.array(nose_bot_corner_pts[1]), np.array(low_pt))
        pas.append(pointy_angle)

        bot_pts = [(114, 343), (189, 413), (128, 357)]
        bot_pts_fn = [get_point3_fn(pt, res, detection) for pt in bot_pts]
        bot_pts_dists = [get_dist_2pts(pt[0], pt[1]) for pt in bot_pts_fn]
        bot_width_mean = np.mean(bot_pts_dists)
        nws.append(bot_width_mean)


        bridge_pts = [(196, 419), (3, 248), (122,351)]
        bridge_pts_fn = [get_point3_fn(pt, res, detection) for pt in bridge_pts]
        bridge_pts_dists = [get_dist_2pts(pt[0], pt[1]) for pt in bridge_pts_fn]
        bridge_width_mean = np.mean(bridge_pts_dists)
        bws.append(bridge_width_mean)



    #     print(get_point3([48, 331], res=res))
    #     print(image.shape)
    #     plt.imshow(image)

        Nose_Bridge_Up_Dn = 82.6
        Nose_Bridge_Fr_Bk = 31.9
        M_Nose_Fr = 26.3
        R_Nose_Con_Out = -20 + (57+20) * (nose_bot_len-0.22) / (0.32 - 0.22)
        L_Nose_Con_Out = R_Nose_Con_Out
        L_Nose_Con_Up = 32.7 + (95.6 - 32.7)*(pointy_angle - 0.01)/(0.08-0.01)
        R_Nose_Con_Up = L_Nose_Con_Up
        Nose_Bridge_Side_Scale = 36.3 + (bot_width_mean - 0.14)*(86.2 - 36.3)/(0.23-0.14)
        Nose_Bridge_Up_Dn = Nose_Bridge_Side_Scale * 3
        Nose_Bridge_Fr_Bk =  -Nose_Bridge_Side_Scale * 1.5


        nose_len_pts = [2, 8]
        nose_len_pts_fn = get_point3_fn(nose_len_pts, res, detection)
        nose_len = get_dist_2pts(nose_len_pts_fn[0], nose_len_pts_fn[1])
        noslens.append(nose_len)
        M_Nose_Up = -45 + (-80+45)*(nose_len-0.32)/(0.46 - 0.32)
#         print( nose_bot_len,pointy_angle, bridge_width_mean, bot_width_mean)

        out_data.append([img_path.split('/')[-1], Nose_Bridge_Up_Dn, Nose_Bridge_Fr_Bk, M_Nose_Fr, R_Nose_Con_Out, L_Nose_Con_Out, L_Nose_Con_Up, R_Nose_Con_Up, Nose_Bridge_Side_Scale, M_Nose_Up])
#         time.sleep(.5)
    blend_shapes_params = pd.DataFrame(out_data, columns=['img_name','Nose_Bridge_Up_Dn', 'Nose_Bridge_Fr_Bk', 'M_Nose_Fr', 'R_Nose_Con_Out', 'L_Nose_Con_Out', 'L_Nose_Con_Up', 'R_Nose_Con_Up', 'Nose_Bridge_Side_Scale', 'M_Nose_Up' ])
    return blend_shapes_params

        
