import numpy as np


def normalize_image(image):
    """
    normalizing an image (from the image [min;max] to [0;1])
    :param image: input image
    :return: normalized image
    """
    max_image = np.amax(image)
    min_image = np.amin(image)
    image = np.asarray(image, np.float32)

    image = (image - min_image)/(max_image - min_image)
    return image


def normalize_image_min_max(image, min_image, max_image):
    """
    normalizing an image (from [minImage;maxImage] to [0;1])
    :param image: input image
    :param min_image:
    :param max_image:
    :return: normalized image
    """
    image = np.asarray(image, np.float32)

    image = (image - min_image)/(max_image - min_image)
    return image
