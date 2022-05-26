import numpy as np
from sklearn.cluster import KMeans


class ColorExtractor:

    def convert_rgb2hex(self, color):

        hex_color = ""
        for i in range(len(color)):
            if color[i] < 16:
                hex_color += '0' + hex(int(color[i])).upper().lstrip("0X")
            else:
                hex_color += hex(int(color[i])).upper().lstrip("0X")

        return "#" + hex_color

    def remove_black_white_points(self, image):

        dummy_image = image.copy()

        mask = np.logical_and.reduce((dummy_image[:, :, 0] > 0, dummy_image[:, :, 1] > 0, dummy_image[:, :, 2] > 0,
                                      dummy_image[:, :, 0] < 256, dummy_image[:, :, 1] < 256,
                                      dummy_image[:, :, 2] < 256))

        return np.array(dummy_image[mask])

    def get_dominant_color_hex(self, image):
        color_list = self.remove_black_white_points(image)
        color_rgb = self.get_dominant_cluster_color(color_list)

        if color_rgb is None:
            return None

        color_rgb = np.floor(color_rgb)
        color_hex = self.convert_rgb2hex(color_rgb)
        return color_hex

    def get_dominant_cluster_color(self, image_color_rgb_list):

        if len(image_color_rgb_list) < 3:
            return None

        cluster = KMeans(n_clusters=3, n_init=3)
        cluster.fit(image_color_rgb_list)

        cluster_centers = cluster.cluster_centers_
        cluster_labels = cluster.labels_

        max_cluster_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_dominant_color_rgb = cluster_centers[np.argmax(counts)]

        return cluster_dominant_color_rgb