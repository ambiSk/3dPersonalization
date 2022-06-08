#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import dlib
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import math
import argparse


# # Load the model

# model_path = 'faceshape_trial_v5.pkl'
# loaded_model = pickle.load(open(model_path, 'rb'))
# model_path = input('Enter the model path  ')
# if(model_path):
#     print(model_path)
#     loaded_model = pickle.load(open(model_path, 'rb'))
# # result = loaded_model.score(x, y)
# # print(result)
# # print(loaded_model.predict(np.array(df3.iloc[1]).reshape(1, -1)))


# # # Testing on the human face

# # In[14]:

# #Input Directory path
# real_image_dir = '/Users/srishtigoel/Downloads/raw_image_female/raw_image/*'

# path = input('Enter the input directory path  ')
# # print(path)

# if(path):

#     real_image_dir = path + '/*'
#     # real_image_dir = '/Users/srishtigoel/Downloads/imagesForDemo/ri_*.png'
#     # real_image_dir = '/Users/srishtigoel/Downloads/raw_image_female/raw_image/*'
# print(real_image_dir)

# In[15]:


import glob
import cv2
import dlib
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#add the dlib path
predictor_path = "/home/sharathchandra/shape_predictor_68_face_landmarks.dat"
from scipy.spatial import ConvexHull, convex_hull_plot_2d
SPLINEPOINTS=50

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# In[16]:


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def get_biggest_bounding_box(bounding_boxes):
    index_of_bounding_box = -1
    max_area = -1
    for i, bounding_box_each in enumerate(bounding_boxes):
        height = bounding_box_each.height()
        width = bounding_box_each.width()
        area = height*width
        if area > max_area:
            area = max_area
            index_of_bounding_box = i
    if index_of_bounding_box == -1:
        return None
    else:
        return index_of_bounding_box
def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M
def correct_pose(image):
    bounding_boxes_all = detector(image, 0)
    bounding_box_index = get_biggest_bounding_box(bounding_boxes_all)
    if bounding_box_index is None:
        return None
    bounding_box = bounding_boxes_all[bounding_box_index]
    face_points = predictor(image, bounding_box)
    left_eye = extract_left_eye_center(face_points)
    right_eye = extract_right_eye_center(face_points)
    M = get_rotation_matrix(left_eye, right_eye)
    w,h,c = image.shape
    image_max_size_possible = int(math.sqrt(w*w + h*h))
    offset_x = int((image_max_size_possible - image.shape[0])/2)
    offset_y = int((image_max_size_possible - image.shape[1])/2)
    dst_image = np.ones((image_max_size_possible, image_max_size_possible, 3),  dtype='uint8') * 255
    dst_image[offset_x:(offset_x + image.shape[0]), offset_y:(offset_y + image.shape[1]), :] = image
    pose_corrected_image = cv2.warpAffine(src=dst_image, dsize=(image_max_size_possible, image_max_size_possible), M=M, borderValue=[255,255,255])
    return pose_corrected_image

def findface(image):
    bounding_boxes_all = detector(image, 1)
    bounding_box_index = get_biggest_bounding_box(bounding_boxes_all)
    if bounding_box_index is None:
        return None
    bounding_box = bounding_boxes_all[bounding_box_index]
    face_points = predictor(image, bounding_box)
    return face_points

def find_hull_im(im):
    shape = findface(im)
    x=[]
    y=[]
    for i in range(2,15):
        x.append(shape.part(i).x)
        y.append(shape.part(i).y)

    x = np.asarray(x)
    y = np.asarray(y)
    dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))

    spline, u = scipy.interpolate.splprep([x, y], u=dist_along, s=0)
    
    # resample it at smaller distance intervals
    interp_d = np.linspace(dist_along[0], dist_along[-1], SPLINEPOINTS)
    interp_x, interp_y = scipy.interpolate.splev(interp_d, spline)
    # plot(interp_x, interp_y, '-o')
    c =[]
    for i in range(len(interp_x)):
        c.append([interp_x[i],interp_y[i]])
    c = np.asarray(c)

    hull1 = ConvexHull(c)
    return hull1,c,shape

def normalize_points(np_array):
    max_point = np.max(np_array, axis=0)
    min_point = np.min(np_array, axis=0)
    height = max_point - min_point
    return (np_array - min_point)/height


# In[17]:



def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


import operator
def find_polyfit_coeff(X,Y,degree):
    
    polynomial_features= PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(X)

    model = LinearRegression(normalize=True)
    model.fit(x_poly,Y)
    # model.score(X, Y)
    y_poly_pred = model.predict(x_poly)


#     plt.scatter(X, Y, s=10)
#     # sort the values of x before line plot
#     sort_axis = operator.itemgetter(0)
#     sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
#     x, y_poly_pred = zip(*sorted_zip)
#     plt.plot(x, y_poly_pred, color='m')
#     plt.show()
    return model.coef_

#     import operator
#     rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
#     r2 = r2_score(Y,y_poly_pred)
#     print(rmse)
#     print(r2)

def find_ratio(shape):
    x=[]
    y=[]
    for i in range(17):
        x.append(shape.part(i).x)
        y.append(shape.part(i).y)
    x = np.asarray(x)
    y = np.asarray(y)


    left = int(min(x))
    right = int(max(x))
    top = int(min(y))
    bottom = int(max(y))
    w = right - left
    h = bottom-top

    h = h + h/1.61
#     face_ratio_x = w/(h*1.5)
    face_ratio_x = h/w
#     face_ratio_x = min(face_ratio_x,1.05)
#     face_ratio_x = max(face_ratio_x,0.96)
    face_ratio_x = round(face_ratio_x, 2)
    
    return face_ratio_x

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((17, 2), dtype=dtype)
    for i in range(0, 17):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def find_angle_list_preset_face_shapes(face_dlib_points):
    list_angle = []
    angle_list =[]
    for i in range(len(face_dlib_points)-1):
        angle = (face_dlib_points[i][1]-face_dlib_points[i+1][1])/(face_dlib_points[i][0]-face_dlib_points[i+1][0])
        angle = np.arctan(angle)
        angle_list.append(angle)
    return angle_list

def get_best_preset_face_shape(face_points_shape):
    face_points_np = shape_to_np(face_points_shape)
    angle_list = find_angle_list_preset_face_shapes(face_points_np)
    return angle_list

def find_dist(p1,p2):
    return ((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)

def main(images_path,model_path):
    print(images_path,model_path)
    loaded_model = pickle.load(open(model_path, 'rb'))
    real_image_dir = images_path + '*'
    image_nm=[]
    fp=[]
    image_name_list=[]
    Side_Face=[]
    cheekup=[]
    jawout=[]
    cheekout=[]
    jawup=[]
    chin=[]
    jawthick=[]
    hup=[]
    hfr=[]
    ckup=[]
    for image_path in glob.glob(real_image_dir):
        try:
            print(image_path)
            im = cv2.imread(image_path)
            image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            image = correct_pose(image)
            if image is None:
                continue
            hull_image,c,facepoints = find_hull_im(image)
            image_face_width_height_ratio = find_ratio(facepoints)
#             if(image_face_width_height_ratio < 1):
#                 image_face_width_height_ratio = (1-image_face_width_height_ratio)*250
#             elif(image_face_width_height_ratio > 1):
#                 image_face_width_height_ratio = (image_face_width_height_ratio-1)*-200
#                 plt.imshow(image)
#                 plt.show()
#             else:
#                 image_face_width_height_ratio=0
#             print(image_face_width_height_ratio)
            if(image_face_width_height_ratio < 1.1):
                image_face_width_height_ratio = (image_face_width_height_ratio-1)*-100
            elif(image_face_width_height_ratio > 1.1):
                image_face_width_height_ratio = (image_face_width_height_ratio-1)*100
        #         plt.imshow(image)
        #         plt.show()
            else:
                image_face_width_height_ratio=0
            print(image_face_width_height_ratio)
            
            xh=[]
            yh=[]
            for i in range(17):
                xh.append(int(facepoints.part(i).x))
                yh.append(int(facepoints.part(i).y))
            xh = np.asarray(xh)
            yh = np.asarray(yh)
            xh = normalize_points(xh)
            yh = normalize_points(yh)
            dist = find_dist([xh[4],yh[4]],[xh[12],yh[12]])

            print('dist: ',dist)
            #interpolate dist in range of -100 to 50 - 0.4-1
            #0.4 -- > -100
            #1   -- >   50
            OldMax=1
            OldMin= 0.4 
            NewMax=50
            NewMin=-100
            OldRange = (OldMax - OldMin)  
            NewRange = (NewMax - NewMin)  
            NewValue = round((((dist - OldMin) * NewRange) / OldRange) + NewMin,2)

            anglelist = get_best_preset_face_shape(facepoints)
            fp=[]
            for i in range(3,12):
                fp.append(anglelist[i])
            print('fp is : ' , fp)
        #     preds = loaded_model.predict(np.array(coeff).reshape(1, -1))[0]
            preds = loaded_model.predict(np.array(fp).reshape(1, -1))[0]
            print(preds)
            #   Side_Face(max)  L_Cheek_Up  L_Cheek_Out L_Jaw_Up    L_Jaw_Out
            image_name_list.append(image_path.split('/')[-1])
            Side_Face.append(round(image_face_width_height_ratio,2))#preds[0])
            cheekup.append(round(preds[1],2))
            jawout.append(round(preds[2],2))
            cheekout.append(round(preds[0],2))
            jawup.append(round(preds[3],2))
            chin.append(round(preds[4],2))
            jawthick.append(NewValue)
        except:
            print('skipping')
            image_name_list.append(image_path.split('/')[-1])
            Side_Face.append(0)#preds[0])
            cheekup.append(0)
            jawout.append(0)
            cheekout.append(0)
            jawup.append(0)
            chin.append(0)
            jawthick.append(0)
    df = pd.DataFrame({'image_name':image_name_list,'Side_Face':Side_Face,'L_Cheek_Up_S':cheekup,'R_Cheek_Up_S':cheekup,'L_Cheek_Out_S':cheekout,'R_Cheek_Out_S':cheekout,'L_Jaw_Up':jawup,'R_Jaw_Up':jawup,'L_Jaw_Out':jawout,'R_Jaw_Out':jawout,'M_Chin_Up':chin,'Jaw_Back_Thick':jawthick})
    
#     df1 = df.drop(['Facial_BS.Head_Fr_Bk','Facial_BS.Head_Up','Facial_BS.L_Cheekbone_Out','Facial_BS.R_Cheekbone_Out'],axis=1)
    df.to_csv('temp.csv')
    
    return df
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter the images path and model path')
    parser.add_argument('--images_path', metavar='path', required=True,
                        help='the path to images folder')
    parser.add_argument('--model_path', metavar='path', required=True,
                        help='path to model')
    args = parser.parse_args()
    
    df = main(args.images_path,args.model_path)
#     print(df)
#     return df





