import torch
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms.functional as F

_MAX_LEVEL = 10.0

class HsbColorAugmenter:
    def __init__(self, hue_sigma_range, saturation_sigma_range, brightness_sigma_range):
        self.hue_sigma_range = hue_sigma_range
        self.saturation_sigma_range = saturation_sigma_range
        self.brightness_sigma_range = brightness_sigma_range

    def randomize(self):
        self.hue_sigma = random.uniform(*self.hue_sigma_range)
        self.saturation_sigma = random.uniform(*self.saturation_sigma_range)
        self.brightness_sigma = random.uniform(*self.brightness_sigma_range)

    def transform(self, image):
        image = image.astype(np.float32) / 255.0
        image_hsv = np.copy(image)
        image_hsv[..., 0] += self.hue_sigma
        image_hsv[..., 1] *= (1 + self.saturation_sigma)
        image_hsv[..., 2] *= (1 + self.brightness_sigma)
        image_hsv = np.clip(image_hsv, 0, 1)
        return (image_hsv * 255).astype(np.uint8)

class HedColorAugmenter:
    def __init__(self, haematoxylin_sigma_range, haematoxylin_bias_range,
                 eosin_sigma_range, eosin_bias_range,
                 dab_sigma_range, dab_bias_range,
                 cutoff_range):
        self.haematoxylin_sigma_range = haematoxylin_sigma_range
        self.haematoxylin_bias_range = haematoxylin_bias_range
        self.eosin_sigma_range = eosin_sigma_range
        self.eosin_bias_range = eosin_bias_range
        self.dab_sigma_range = dab_sigma_range
        self.dab_bias_range = dab_bias_range
        self.cutoff_range = cutoff_range

    def randomize(self):
        self.haematoxylin_sigma = random.uniform(*self.haematoxylin_sigma_range)
        self.haematoxylin_bias = random.uniform(*self.haematoxylin_bias_range)
        self.eosin_sigma = random.uniform(*self.eosin_sigma_range)
        self.eosin_bias = random.uniform(*self.eosin_bias_range)
        self.dab_sigma = random.uniform(*self.dab_sigma_range)
        self.dab_bias = random.uniform(*self.dab_bias_range)

    def transform(self, image):
        image = image.astype(np.float32) / 255.0
        image[..., 0] = image[..., 0] * (1 + self.haematoxylin_sigma) + self.haematoxylin_bias
        image[..., 1] = image[..., 1] * (1 + self.eosin_sigma) + self.eosin_bias
        image[..., 2] = image[..., 2] * (1 + self.dab_sigma) + self.dab_bias
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

def hsv(image, factor):
    image = np.transpose(np.array(image), (2, 0, 1))
    augmentor = HsbColorAugmenter(
        hue_sigma_range=(-factor, factor),
        saturation_sigma_range=(-factor, factor),
        brightness_sigma_range=(0, 0)
    )
    augmentor.randomize()
    return Image.fromarray(np.transpose(augmentor.transform(image), (1, 2, 0)))

def hed(image, factor):
    image = np.transpose(np.array(image), (2, 0, 1))
    augmentor = HedColorAugmenter(
        haematoxylin_sigma_range=(-factor, factor),
        haematoxylin_bias_range=(-factor, factor),
        eosin_sigma_range=(-factor, factor),
        eosin_bias_range=(-factor, factor),
        dab_sigma_range=(-factor, factor),
        dab_bias_range=(-factor, factor),
        cutoff_range=(0.15, 0.85)
    )
    augmentor.randomize()
    return Image.fromarray(np.transpose(augmentor.transform(image), (1, 2, 0)))

def equalize(image):
    image = Image.fromarray(image.astype(np.uint8))
    return ImageOps.equalize(image)


def identity(image):
    """Implements Identity

    """
    return image


def distort_image_with_randaugment(image, num_ops=1, magnitude=2):
    ops = list(NAME_TO_FUNC.keys())
    for _ in range(num_ops):
        op_name = random.choice(ops)
        op = NAME_TO_FUNC[op_name]
        # if op_name in ["ShearX", "ShearY"]:
        #     level = _shear_level_to_arg(magnitude)[0]
        #     image = op(image, level, replace=(128, 128, 128))
        # elif op_name in ["TranslateX", "TranslateY"]:
        #     level = _translate_level_to_arg(magnitude, image.size[0] * 0.45)[0]
        #     image = op(image, level, replace=(128, 128, 128))
        # elif op_name == "Rotate":
        #     level = _randomly_negate_tensor(magnitude * 30 / _MAX_LEVEL)
        #     image = op(image, level, replace=(128, 128, 128))
        if op_name in ["Color", "Contrast", "Brightness", "Sharpness"]:
            level = _enhance_level_to_arg(magnitude)[0]
            image = op(image, level)
        elif op_name in ['Equalize']:
            image = op(image)
            image = np.array(image)
        elif op_name in ["Posterize"]:
            level = int((magnitude / _MAX_LEVEL) * 4) + 4
            image = op(image, level)
        elif op_name in ["Solarize"]:
            level = int((magnitude / _MAX_LEVEL) * 256)
            image = op(image, level)
        elif op_name in ["Hsv", "Hed"]:
            level = magnitude / _MAX_LEVEL
            image = op(image, level)
            image = np.array(image)
        else:
            image = op(image)
    return image

NAME_TO_FUNC = {
    # "AutoContrast": autocontrast,
    "Hsv": hsv,
    "Hed": hed,
    "Identity": identity,
    "Equalize": equalize,
    # "Invert": invert,
    # "Rotate": rotate,
    # "Posterize": posterize,
    # "Solarize": solarize,
    # "Color": color,
    # "Contrast": contrast,
    # "Brightness": brightness,
    # "Sharpness": sharpness,
    # "ShearX": shear_x,
    # "ShearY": shear_y,
    # "TranslateX": translate_x,
    # "TranslateY": translate_y,
}

def _randomly_negate_tensor(value):
    return value if random.random() > 0.5 else -value

def _enhance_level_to_arg(level):
    return (level / _MAX_LEVEL * 1.8 + 0.1,)

def _shear_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.3
    return (_randomly_negate_tensor(level),)

def _translate_level_to_arg(level, translate_const):
    level = (level / _MAX_LEVEL) * translate_const
    return (_randomly_negate_tensor(level),)
