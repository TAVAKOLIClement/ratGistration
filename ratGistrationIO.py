import os, glob

import numpy as np
import math

import imageio
import fabio
import fabio.edfimage as edf


def create_list_of_files(folder_name, extension):
    """
    creating a list of files with a corresponding extension in an input folder
    :param folder_name: folder name
    :param extension: extension of the target files
    :return: the list of files sorted
    """
    list_of_files = glob.glob(folder_name + '/*' + extension)
    list_of_files.sort()
    return list_of_files


def open_image(filename):
    """
    opening a 2D image
    :param filename: file name
    :return: image
    """
    filename = str(filename)
    im = fabio.open(filename)
    imarray = im.data
    return imarray


def get_header(filename):
    """
    retrieving the header of an image
    :param filename: file name
    :return: header
    """
    im = fabio.open(filename)
    header = im.header
    return header


def open_seq(filenames):
    """
    opening a sequence of images
    :param filenames: file names
    :return: image
    """
    if len(filenames) > 0:
        data = open_image(str(filenames[0]))
        height, width = data.shape
        to_return = np.zeros((len(filenames), height, width), dtype=np.float32)
        i = 0
        for file in filenames:
            data = open_image(str(file))
            to_return[i, :, :] = data
            i += 1
        return to_return
    raise Exception('spytlabIOError')


def save_edf_image(data, filename):
    """
    saving an image to .edf format
    :param data: input image
    :param filename: filename
    :return: None
    """
    data_to_store = data.astype(np.float32)
    edf.EdfImage(data=data_to_store).write(filename)


def save_edf_image_int(data, filename):
    """
    saving an image to .edf format (int)
    :param data: input image
    :param filename: filename
    :return: None
    """
    data_to_store = data.astype(np.int)
    edf.EdfImage(data=data_to_store).write(filename)


def save_edf_sequence(image, path):
    """
    saving a custom .edf volume
    :param image: input image
    :param path: filename
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(0, image.shape[0]):
        if i < 10:
            save_edf_image(image[i, :, :], path + '000' + str(i) + '.edf')
        else:
            if i < 100:
                save_edf_image(image[i, :, :], path + '00' + str(i) + '.edf')
            else:
                save_edf_image(image[i, :, :], path + '0' + str(i) + '.edf')


def save_edf_sequence_int(image, path):
    """
    saving a custom .edf volume (int)
    :param image: input image
    :param path: filename
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(0, image.shape[0]):
        if i < 10:
            save_edf_image_int(image[i, :, :], path + '000' + str(i) + '.edf')
        else:
            if i < 100:
                save_edf_image_int(image[i, :, :], path + '00' + str(i) + '.edf')
            else:
                save_edf_image_int(image[i, :, :], path + '0' + str(i) + '.edf')


def save_edf_and_crop(image, shape, path):
    """
    saving a custom .edf volume and cropping it
    :param image: input image
    :param shape: shape to crop into
    :param path: filename
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

    y_diff_min = math.floor((image.shape[1] - shape[1]) / 2)
    y_diff_max = image.shape[1] - math.ceil((image.shape[1] - shape[1]) / 2) + 1
    x_diff_min = math.floor((image.shape[2] - shape[2]) / 2)
    x_diff_max = image.shape[2] - math.ceil((image.shape[2] - shape[2]) / 2) + 1

    for i in range(0, image.shape[0]):

        slice_nb = image[i, y_diff_min:y_diff_max, x_diff_min:x_diff_max]

        if i < 10:
            save_edf_image(slice_nb, path + '000' + str(i) + '.edf')
        else:
            if i < 100:
                save_edf_image(slice_nb, path + '00' + str(i) + '.edf')
            else:
                save_edf_image(slice_nb, path + '0' + str(i) + '.edf')


def save_tif_sequence(image, path):
    """
    saving a custom .tif volume
    :param image: input image
    :param path: filename
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(0, image.shape[0]):
        if i < 10:
            imageio.imwrite(path + '000' + str(i) + '.tif', image[i, :, :])
        else:
            if i < 100:
                imageio.imwrite(path + '00' + str(i) + '.tif', image[i, :, :])
            else:
                imageio.imwrite(path + '0' + str(i) + '.tif', image[i, :, :])


def save_tif_sequence_int(image, path):
    """
    saving a custom .tif volume (int)
    :param image: input image
    :param path: filename
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

    image = image.astype(np.int)
    for i in range(0, image.shape[0]):
        if i < 10:
            imageio.imwrite(path + '000' + str(i) + '.tif', image[i, :, :])
        else:
            if i < 100:
                imageio.imwrite(path + '00' + str(i) + '.tif', image[i, :, :])
            else:
                imageio.imwrite(path + '0' + str(i) + '.tif', image[i, :, :])


def save_tif_sequence_and_crop(image, shape, path):
    """
    saving a custom .tif volume and cropping it
    :param image: input image
    :param shape: shape to crop into
    :param path: filename
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

    y_diff_min = math.floor((image.shape[1] - shape[1]) / 2)
    y_diff_max = image.shape[1] - math.ceil((image.shape[1] - shape[1]) / 2) + 1
    x_diff_min = math.floor((image.shape[2] - shape[2]) / 2)
    x_diff_max = image.shape[2] - math.ceil((image.shape[2] - shape[2]) / 2) + 1

    for i in range(0, image.shape[0]):

        slice_nb = image[i, y_diff_min:y_diff_max, x_diff_min:x_diff_max]

        if i < 10:
            imageio.imwrite(path + '000' + str(i) + '.tif', slice_nb)
        else:
            if i < 100:
                imageio.imwrite(path + '00' + str(i) + '.tif', slice_nb)
            else:
                imageio.imwrite(path + '0' + str(i) + '.tif', slice_nb)
