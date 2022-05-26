import os
import cv2
from mtcnn import mtcnn
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import tensorflow as tf
import csv
import shutil

from fastai.vision.all import *
import torch
from PIL import Image, ImageDraw, ImageFont
#%matplotlib inline


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.get_logger().setLevel('ERROR')

"""
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[2:3], 'GPU')
    torch.cuda.set_device('cuda:2')
    tf.config.experimental.set_memory_growth(physical_devices[2], True)
except:
    pass
"""

class MTCNN:
    def __init__(self):
        self.detector = mtcnn.MTCNN()

    def get_cropped_image(self, img_in, img_out, crop_forehead=True):
        image = cv2.cvtColor(cv2.imread(img_in), cv2.COLOR_BGR2RGB)
        result = self.detector.detect_faces(image)
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']
        confidence = result[0]['confidence']
        if confidence > 0.5:
            y_new = max(keypoints['left_eye'][1], keypoints['right_eye'][1])
            if crop_forehead:
                image = image[bounding_box[1]: y_new, bounding_box[0]: bounding_box[0] + bounding_box[2], ]
            else:
                image = image[y_new: bounding_box[1] + bounding_box[3],
                        bounding_box[0]: bounding_box[0] + bounding_box[2], ]
            cv2.imwrite(img_out, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def plot_image(self, img_in):
        img = mpimg.imread(img_in)
        plt.imshow(img)
        plt.show()


#mtcnn_obj = MTCNN()

model_path = "../../Trained_models/cheeklines_v1.pkl"
if not os.path.exists(model_path):
    os.system("gsutil cp gs://ds-repository/3D-Hikemoji-pers/trained-models/cheeklines_v1.pkl " + model_path)

#learn = load_learner(model_path)

# fnt = ImageFont.truetype(font='/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', size=20)


def run_inference(images_path, csv_file=None, copy_dest_path=None):
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('displaying gpussssss ',tf.config.list_physical_devices('GPU'))
    #with tf.device('/gpu:1'):
    mtcnn_obj = MTCNN()
    #with torch.cuda.device(1):
    learn = load_learner(model_path)
    tmp_path = "/tmp/tmp-dir-infer"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.mkdir(tmp_path)

    if os.path.isfile(images_path):
        img_paths = [images_path, ]
    else:
        img_paths = [os.path.join(images_path, img_path) for img_path in os.listdir(images_path)]

    if csv_file is not None:
        os.makedirs(os.path.join(csv_file.split("/")[-1]), exist_ok=True)
        out_file = open(csv_file, "w")
        csvwriter = csv.writer(out_file)
    if copy_dest_path is not None:
        os.makedirs(copy_dest_path, exist_ok=True)
        fnt = ImageFont.truetype(font='/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', size=20)

    for src_ in img_paths:
        if not (src_.endswith("png") or src_.endswith("jpg") or src_.endswith("jpeg")):
            continue
        tmp_dest = os.path.join(tmp_path, src_.split("/")[-1])

        mtcnn_obj.get_cropped_image(src_, tmp_dest, crop_forehead=False)
        pred = learn.predict(tmp_dest)[1].bool().numpy()[0]

        if copy_dest_path is not None:
            qr = Image.open(src_)
            background = Image.new('RGB', (qr.size[0], qr.size[1] + 50), (255, 255, 255, 255))
            draw = ImageDraw.Draw(background)

            text = "Pred:" + str(pred)
            draw.text((5, 5), text, (0, 0, 0), spacing=20, stroke_width=0, font=fnt)
            background.paste(qr, (0, 50))

            final_dest = os.path.join(copy_dest_path, src_.split("/")[-1])

            background.save(final_dest)

        if csv_file is not None:
            csvwriter.writerow([src_.split("/")[-1], str(pred)])

        print("Done", src_)

    if csv_file:
        out_file.close()

    shutil.rmtree(tmp_path, ignore_errors=True)
