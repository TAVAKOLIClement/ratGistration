import numpy as np

from skimage.transform import resize

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


def bin_resize(image, bin_factor):
    nb_slices, width, height = image.shape
    if bin_factor > 0:
        nb_slices = int(nb_slices/bin_factor)
        width = int(width/bin_factor)
        height = int(height/bin_factor)
        dim = (nb_slices, width, height)
        return resize(image,dim, preserve_range=True)
    raise Exception('bin_factor must be strictly positive')