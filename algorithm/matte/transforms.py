import math
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images


class ResizeIfBiggerThan(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images):
        for idx, image in enumerate(images):
            max_size = max(image.size)
            if max_size > self.size:
                rate = self.size / float(max_size)
                h, w = math.ceil(rate * image.size[0]), math.ceil(rate * image.size[1])
                images[idx] = F.resize(image, [w, h])
        return images


class ToTensor(object):
    def __call__(self, images):
        for idx, image in enumerate(images):
            images[idx] = F.to_tensor(image)
        return images
