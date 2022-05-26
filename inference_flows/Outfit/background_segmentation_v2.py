import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
from gevent import lock


class BackgroundSegmentation(object):
    _instance = None
    _lock = lock.RLock()

    __version = "bg_seg_w_v2"

    def get_version(cls):
        return cls.__version

    frozen_graph_name = "frozen_inference_graph"
    input_tensor_name = 'ImageTensor:0'
    output_tensor_name = 'SemanticPredictions:0'

    INPUT_TENSOR_SIZE = 513
    model_path = "Pretrained_models/background_segmentation/mobile_net_model/frozen_inference_graph.pb"

    flag_inference_usable = 1

    def __init__(self):

        self.graph = tf.Graph()
        graph_def = tf.GraphDef.FromString(open(self.model_path, "rb").read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=self.config)

    @classmethod
    def get_instance(cls):
        if not BackgroundSegmentation._instance:
            with cls._lock:
                if not BackgroundSegmentation._instance:
                    BackgroundSegmentation._instance = BackgroundSegmentation()

        return BackgroundSegmentation._instance

    @classmethod
    def preprocess_image(cls, image):
        # EXPECTS BINARY IMAGE FILE
        try:
            image_width, image_height = image.size
            resize_ratio = 1.0 * cls.INPUT_TENSOR_SIZE / max(image_width, image_height)
            target_size = (int(resize_ratio * image_width), int(resize_ratio * image_height))
            resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
            return resized_image, resized_image.size
        except Exception:
            return None, None

    def run_inference(self, resized_image, timer_prefix=None):
        segmentation_map = self.sess.run(
            self.output_tensor_name,
            feed_dict={self.input_tensor_name: [np.asarray(resized_image)]}
        )

        return segmentation_map[0]

    @staticmethod
    def overlay_segmentation_on_image(resized_image, segmentation_map, timer_prefix=None):

        if segmentation_map is None:
            return resized_image

        i, j = np.where(segmentation_map == 0)
        base_img = np.asarray(resized_image)

        # Added copy because of this. Have to verify once for server images.
        # ValueError: assignment destination is read-only

        segmented_image = base_img.copy()
        segmented_image[i, j, :] = 255
        return segmented_image

    def get_background_segmented_image(self, image):
        resized_image, resized_image_size = self.preprocess_image(image)
        segmentation_map = self.run_inference(resized_image)
        return self.overlay_segmentation_on_image(resized_image, segmentation_map)

    def get_background_segmentation_map(self, image):
        resized_image, resized_image_size = self.preprocess_image(image)
        segmentation_map = self.run_inference(resized_image)
        return segmentation_map
