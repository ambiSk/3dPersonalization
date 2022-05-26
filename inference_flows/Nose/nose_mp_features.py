#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
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


# In[3]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/mnt/mujtaba/vision/nose/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)


# In[4]:


data_dir = '/mnt/mujtaba/vision/nose/data/female_raw/'
# data_dir = '/nfs_storage/fs-mnt6/mujtaba/3D_vision/Nose/results/female_v0_out/'
img_files = [data_dir +'/'+filenm for filenm in os.listdir(data_dir)]


# In[5]:



def get_detection_bb(detection):
    return (detection.location_data.relative_bounding_box.xmin, detection.location_data.relative_bounding_box.ymin, 
detection.location_data.relative_bounding_box.width, detection.location_data.relative_bounding_box.height)

def face_normalize(point_ind, res_landmark, detection):
    dbb = get_detection_bb(detection)
    bb_tl = (dbb[0], dbb[1])
    bb_wh = (dbb[2], dbb[3])
    pt = (res_landmark[point_ind].x, res_landmark[point_ind].y)
    pt_fn = ((pt[0]-bb_tl[0]) / bb_wh[1] , (pt[1]-bb_tl[1]) / bb_wh[1], res_landmark[point_ind].z)
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
    n_pl_pts = get_point3_fn([357, 189, 413 ], res, detection)
    plane_vec = get_normal_vec_plane(n_pl_pts[0], n_pl_pts[1], n_pl_pts[2])
    proj_hts = [proj_u_over_plane(htpt, plane_vec) for htpt in ht_pts]
#     mean_nose_ht = np.mean([get_dist_2pts(proj_hts[ii], ht_pts[ii]) for ii in range(len(ht_pts))])
    ht_pts = [[0,0,ptt[2]] for ptt in ht_pts]
    mean_nose_ht = np.mean([get_dist_2pts( [0,0,0] , ht_pts[ii]) for ii in range(len(ht_pts))])
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
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    results_fd = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection = results_fd.detections[0]
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        min_detection_confidence=0.5)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    res = results.multi_face_landmarks[0]
    return res, detection


# In[ ]:





# In[ ]:





# In[7]:


def get_image_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res, detection = get_mp_res(image)
    nose_bot_corner_pts = get_point3_fn([48, 331], res, detection)
    face_width_pts = get_point3_fn([ 123, 352], res, detection)
    nose_bot_len = get_dist_2pts(nose_bot_corner_pts[0], nose_bot_corner_pts[1])
    face_width =  get_dist_2pts(face_width_pts[0], face_width_pts[1])
    nose_w = nose_bot_len / face_width
    
    nose_top_pts = get_point3_fn([189, 413 ], res, detection)
    nose_top_wid = get_dist_2pts(nose_top_pts[0], nose_top_pts[1])
    nose_top_w = nose_top_wid / face_width
    

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
    bridge_pts_fn = [get_point3_fn(pt, res, detection) for pt in bridge_pts]
    bridge_pts_dists = [get_dist_2pts(pt[0], pt[1]) for pt in bridge_pts_fn]
    bridge_width_mean = np.mean(bridge_pts_dists) / face_width

    bot_ht, mid_ht, top_ht = get_nose_ht(res, detection, 'bot'), get_nose_ht(res, detection, 'mid'), get_nose_ht(res, detection, 'top')
    brg_top_ratio = mid_ht / top_ht

    nose_len_pts = [2, 8]
    lip_top_pt = [0]
    nose_len_pts_fn = get_point3_fn(nose_len_pts, res, detection)
    lip_pt_fn = get_point3_fn(lip_top_pt, res, detection)
    nose_len = get_dist_2pts(nose_len_pts_fn[0], nose_len_pts_fn[1])
    nose_lip_len = get_dist_2pts(lip_pt_fn[0], nose_len_pts_fn[1])
    nose_len = nose_len / nose_lip_len
    
    nose_vol = 0.25*nose_bot_len*bot_ht*nose_len + 0.25*nose_top_w*top_ht*nose_len    
    
    feat_array = [nose_len, nose_lip_len, nose_w, nose_bot_len, brg_top_ratio, bot_ht, mid_ht, top_ht, bridge_width_mean, 
    bot_width_mean, pointy_angle, pointy_angle1, face_width, nose_vol]
    
    
    return feat_array

