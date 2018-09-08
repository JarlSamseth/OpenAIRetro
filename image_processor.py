import numpy as np


class ImageProcessor:

    def __init__(self):
        pass

    def to_grayscale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)

    def downsample(self, img):
        return img[::2, ::2]

    def preprocess(self, img):
        return self.to_grayscale(self.downsample(img))
