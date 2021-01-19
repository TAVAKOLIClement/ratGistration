import numpy as np

# -- registration library --
import SimpleITK as Sitk

import PyIPSDK.IPSDKIPLBinarization as Bin

import math

# -- 2D Convex Hull function --
import ratGistrationIO
from skimage.measure import label, regionprops


def sum_list_of_vectors(list_of_vectors):
    """
    sums all the vectors of a list
    :param list_of_vectors: input list
    :return: resulting vector
    """
    final_vector = np.zeros(list_of_vectors[0].shape)
    for vector in list_of_vectors:
        final_vector += vector

    return final_vector


def apply_2d_rotation_to_a_vector(vector, angle):
    """
    just apply a 2d rotation to a vector depending on an input angle (anticlockwise)
    :param vector: 2d numpy vector
    :param angle: angle in degrees
    :return: rotated vector
    """
    c = math.cos(float(angle))
    s = math.sin(float(angle))

    return np.array([vector[0] * c - vector[1] * s,
                    vector[0] * s + vector[1] * c])


def compute_2d_rotation(image, angle, interpolator_type="linear"):
    """
    computes a 2d rotation on an image based on an angle around axis z
    :param image: input image
    :param angle: angle
    :param interpolator_type: type of interpolator (linear or nearest neighbor)
    :return: rotated image
    """
    image_itk = Sitk.GetImageFromArray(image)
    tx = Sitk.AffineTransform(image_itk.GetDimension())

    c = math.cos(float(angle))
    s = math.sin(float(angle))
    tx.SetMatrix((c, s, 0,
                  -s, c, 0,
                  0, 0, 1))

    tx.SetCenter((image.shape[2] / 2, image.shape[1] / 2, 0))

    if interpolator_type == "linear":
        interpolator = Sitk.sitkLinear
    else:
        interpolator = Sitk.sitkNearestNeighbor

    image_itk = Sitk.Resample(image_itk, tx, interpolator, 0.0, image_itk.GetPixelIDValue())

    resulting_image = Sitk.GetArrayFromImage(image_itk)

    return resulting_image


def calculate_rotation_matrix_between_3d_vectors(current_vector, target_vector):
    """
    computes the rotation matrix in order to align current_vector on target_vector (x, y, z)
    :param current_vector: vector we're aligning
    :param target_vector: vector we're aligning on
    :return: rotation matrix
    """

    current_vector = current_vector / np.linalg.norm(current_vector)
    target_vector = target_vector / np.linalg.norm(target_vector)

    c = np.dot(current_vector, target_vector)  # [x, y, z]
    l = np.cross(current_vector, target_vector)  # [x, y, z]
    s = np.linalg.norm(l)

    # Rotation matrix
    kmat = np.array([[0, -l[2], l[1]], [l[2], 0, -l[0]], [-l[1], l[0], 0]])
    matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return matrix


def compute_3d_rotation(image, rotation_matrix, center_of_rotation, translation=[0, 0, 0]):
    """
    computes rotation arounda center_of_rotation absed on a rotation matrix
    :param image: input image
    :param rotation_matrix: rotation matrix
    :param center_of_rotation: center of rotation (3D, [x, y, z]
    :param translation: 3D translation in addition to the angular rotation
    :return: rotated image
    """
    image_itk = Sitk.GetImageFromArray(image)

    tx = Sitk.AffineTransform(image_itk.GetDimension())

    tx.SetMatrix((rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0],
                  rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[2][1],
                  rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2]))

    tx.SetCenter((center_of_rotation[0], center_of_rotation[1], center_of_rotation[2]))  # [x, y, z]

    tx.SetTranslation((translation[0], translation[1], translation[2]))

    image_itk = Sitk.Resample(image_itk, tx, Sitk.sitkLinear, 0.0, image_itk.GetPixelIDValue())

    image = Sitk.GetArrayFromImage(image_itk)

    return np.copy(image)


def retrieve_throat_centroid(mask):
    """
    Calculates the barycenter of a shape
    :param mask: mask of the shape
    :return: its barycenter
    """
    label_img = label(mask)
    regions = regionprops(label_img)
    centroid = regions[0].centroid
    return centroid


def count_the_needed_translation_for_black_slices(image):
    """
    looks at a rotated image in order to count the number of slices that become black.
    :param image: input rotated image
    :return: the number of slices that contain too much blackness
    """
    offset = 0
    for slice_nb in range(0, image.shape[0]):
        slice_of_image = image[slice_nb, :, :]
        if len(slice_of_image[slice_of_image == 0]) > len(slice_of_image) * 0.4:
            offset += 1
    return offset


def straight_triangle_rotation(image, skull, skull_bounding_box, baryctr_jaw_one, baryctr_jaw_two):
    """
    Uses the position of the jaws/cranial skull to make the rat straight (jaws at the bottom, skull at the top)
    :param skull: mask of the segmented cranial skull
    :param image: input image
    :param skull_bounding_box: skull bounding box
    :param baryctr_jaw_one: barycenter of the first jaw
    :param baryctr_jaw_two: barycenter of the other jaw
    :return:
    """
    skull_center = [skull_bounding_box[0] + (skull_bounding_box[1] - skull_bounding_box[0]) / 2, skull_bounding_box[2] + (skull_bounding_box[3] - skull_bounding_box[2]) / 2]

    if baryctr_jaw_one[0] < baryctr_jaw_two[0]:
        jaws_vector = np.array([baryctr_jaw_two[0] - baryctr_jaw_one[0], baryctr_jaw_two[1] - baryctr_jaw_one[1]])
    else:
        jaws_vector = np.array([baryctr_jaw_one[0] - baryctr_jaw_two[0], baryctr_jaw_one[1] - baryctr_jaw_two[1]])

    target_vector = np.array([np.linalg.norm(jaws_vector), 0])
    if jaws_vector[1] > 0:
        angle = -math.acos(np.dot(target_vector, jaws_vector)/(np.linalg.norm(jaws_vector)*np.linalg.norm(target_vector)))
    else:
        angle = math.acos(np.dot(target_vector, jaws_vector)/(np.linalg.norm(jaws_vector)*np.linalg.norm(target_vector)))

    skull_to_center_vector = np.array([skull_center[0] - image.shape[2]/2,
                                       skull_center[1] - image.shape[1]/2])
    new_skull_to_center_vector = apply_2d_rotation_to_a_vector(skull_to_center_vector, angle)

    jaw_one_to_center_vector = np.array([baryctr_jaw_one[0] - image.shape[2] / 2,
                                         baryctr_jaw_one[1] - image.shape[1] / 2])
    new_jaw_one_to_center_vector = apply_2d_rotation_to_a_vector(jaw_one_to_center_vector, angle)

    if new_jaw_one_to_center_vector[1] < new_skull_to_center_vector[1]:
        angle += math.pi

    resulting_image = compute_2d_rotation(image, angle, "linear")
    resulting_skull = compute_2d_rotation(skull, angle, "nearest")

    return np.copy(resulting_image), np.copy(resulting_skull), float(angle)


def straight_throat_rotation(image, throat_mask_img):
    """
    rotates the image based on the segmentation of the throat (so that is is aligned with a [0, 0, 1] vector
    :param image: input image
    :param throat_mask_img: input throat segmentation
    :return: the aligned image, the rotation matrix and the center of rotation plus the offset (due to the rotation)
    """

    centroid_list = []
    vectors_list = []
    z_length = 0

    for nbSlice in range(0, throat_mask_img.shape[0]):
        z_length += 1
        nb_pixels_throat = np.sum(throat_mask_img[nbSlice, :, :])
        if nb_pixels_throat >= 1:
            centroid = retrieve_throat_centroid(throat_mask_img[nbSlice, :, :])
            centroid_list.append(centroid)
            if nbSlice > 0:
                vectors_list.append(np.array([z_length,
                                              centroid_list[-1][0] - centroid_list[-2][0],
                                              centroid_list[-1][1] - centroid_list[-2][1]]))# vector : [z, y, x]
            z_length = 0

    total_vector = sum_list_of_vectors(vectors_list)
    normalized_total_vector = total_vector/np.linalg.norm(total_vector)
    normalized_current_vector = vectors_list[0]/np.linalg.norm(vectors_list[0])

    while np.dot(normalized_total_vector, normalized_current_vector) < 0.92:
        del vectors_list[0]
        total_vector = sum_list_of_vectors(vectors_list)
        normalized_total_vector = total_vector/np.linalg.norm(total_vector)
        normalized_current_vector = vectors_list[0] / np.linalg.norm(vectors_list[0])

    total_vector = sum_list_of_vectors(vectors_list)
    normalized_total_vector = total_vector/np.linalg.norm(total_vector)
    normalized_current_vector = vectors_list[-1]/np.linalg.norm(vectors_list[-1])

    while np.dot(normalized_total_vector, normalized_current_vector) < 0.92:
        del vectors_list[-1]
        total_vector = sum_list_of_vectors(vectors_list)
        normalized_total_vector = total_vector/np.linalg.norm(total_vector)
        normalized_current_vector = vectors_list[-1] / np.linalg.norm(vectors_list[-1])

    normalized_total_vector_itk = np.flip(np.copy(normalized_total_vector))
    print("Normalized total vector: ", normalized_total_vector)
    print("Flipped vector :", normalized_total_vector_itk)

    center_of_rotation = [centroid_list[0][1], centroid_list[0][0], 0] # [x, y]

    rotation_matrix = calculate_rotation_matrix_between_3d_vectors(normalized_total_vector_itk, np.array([0, 0, 1]))

    test_image = compute_3d_rotation(image, rotation_matrix, center_of_rotation, [0, 0, 0])
    offset = count_the_needed_translation_for_black_slices(test_image)

    image = compute_3d_rotation(image, rotation_matrix, center_of_rotation, [0, 0, offset])
    return np.copy(image), rotation_matrix, center_of_rotation, offset


def symmetry_based_registration(image, skull, skull_bounding_box, throat_coordinates, number_of_iterations):
    """
    Looking for the best rotation (around axis Z) making the skull symmetric
    :param image: input Image
    :param skull: input Image skull mask
    :param skull_bounding_box: skull bounding box (2D)
    :param throat_coordinates: coordinates of the segmented throat
    :param number_of_iterations: how many angles we're trying to check (0.5 degrees step)
    :return: rotated image, rotated_skull
    """

    # The bounding box needs to be centered on the throat coordinates
    if throat_coordinates[1] - skull_bounding_box[0] > skull_bounding_box[1] - throat_coordinates[1]:
        skull_bounding_box[1] = int(skull_bounding_box[0] + (throat_coordinates[1] - skull_bounding_box[0]) * 2)
    else:
        skull_bounding_box[0] = int(skull_bounding_box[1] - (skull_bounding_box[1] - throat_coordinates[1]) * 2)

    cross_correlation_list = []
    increment = 0
    diff = 1

    correct_angle = 0
    for i in range(0, number_of_iterations*2):

        angle = float(-number_of_iterations + i) / 180 * math.pi

        image_copy = np.copy(image)
        skull_copy = np.copy(skull)

        resulting_image = compute_2d_rotation(image_copy, angle, "linear")
        resulting_skull = compute_2d_rotation(skull_copy, angle, "nearest")

        cropped_image = resulting_image[:, skull_bounding_box[2]:skull_bounding_box[3] + 1, skull_bounding_box[0]:skull_bounding_box[1] + 1]
        flipped_image = cropped_image[:, :, ::-1]
        right_half_image = np.copy(flipped_image[:, :, int(flipped_image.shape[2] / 2):flipped_image.shape[2]])
        left_half_image = cropped_image[:, :, int(cropped_image.shape[2] / 2):cropped_image.shape[2]]

        cropped_skull = resulting_skull[:, skull_bounding_box[2]:skull_bounding_box[3] + 1, skull_bounding_box[0]:skull_bounding_box[1] + 1]
        flipped_skull = cropped_skull[:, :, ::-1]
        left_half_skull = np.copy(flipped_skull[:, :, int(flipped_skull.shape[2] / 2):flipped_skull.shape[2]]) # NEED TO CHECK
        right_half_skull = cropped_skull[:, :, int(cropped_skull.shape[2] / 2):cropped_skull.shape[2]]

        left_half_skull[left_half_image == 0] = 0
        right_half_skull[right_half_image == 0] = 0

        number_of_zeros = np.zeros(left_half_skull.shape)
        number_of_zeros[right_half_image == 0] = 1
        number_of_zeros[left_half_image == 0] = 1

        subtraction = right_half_skull - left_half_skull
        count = len(subtraction[subtraction != 0])

        normalized_value = count / len(number_of_zeros[number_of_zeros == 0])

        cross_correlation_list.append(normalized_value)

        if diff > normalized_value:
            correct_angle = angle
            diff = normalized_value

        print("angle : ", angle, "; metric :", normalized_value)
        increment += 1

    image_copy = np.copy(image)
    skull_copy = np.copy(skull)

    print("Best angle :", correct_angle)

    resulting_image = compute_2d_rotation(image_copy, correct_angle, "linear")
    resulting_skull = compute_2d_rotation(skull_copy, correct_angle, "nearest")

    cropped_image = resulting_image[:, skull_bounding_box[2]:skull_bounding_box[3] + 1, skull_bounding_box[0]:skull_bounding_box[1] + 1]
    cropped_skull = resulting_skull[:, skull_bounding_box[2]:skull_bounding_box[3] + 1, skull_bounding_box[0]:skull_bounding_box[1] + 1]

    return np.copy(cropped_image), np.copy(cropped_skull), correct_angle


def command_iteration(method):
    """
    registration verbose output function
    :param method: registration method
    :return: None
    """
    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),
                                     method.GetMetricValue()))
    # print("Position = ", method.GetOptimizerPosition())


def registration_computation_with_mask(moving_image, reference_image, moving_mask, reference_mask, is_rotation_needed=False, verbose=False):
    """
    registration calculation based on a mask
    :param moving_image: image to register
    :param reference_image: reference image
    :param moving_mask: moving image calculation mask
    :param reference_mask: reference image calculation mask
    :param is_rotation_needed: do we compute an additional rotation transformation
    :param verbose: do we print each iteration
    :return: transformations (only translation or both translation + rotation depending on is_rotation_needed)
    """

    # Conversion into ITK format images
    fixed_image_itk = Sitk.GetImageFromArray(reference_image.data)
    moving_image_itk = Sitk.GetImageFromArray(moving_image.data)

    fixed_mask_itk = Sitk.GetImageFromArray(reference_mask.data)
    moving_mask_itk = Sitk.GetImageFromArray(moving_mask.data)

    # --------------------------------------------------
    # ------------ INITIAL TRANSLATION PART ------------
    # --------------------------------------------------

    # Start of registration declaration
    translation_registration_method = Sitk.ImageRegistrationMethod()

    # 1 ---> METRIC
    translation_registration_method.SetMetricAsCorrelation()
    # translation_registration_method.SetMetricAsANTSNeighborhoodCorrelation(2)
    # translation_registration_method.SetMetricAsJointHistogramMutualInformation()
    # translation_registration_method.SetMetricAsMeanSquares()

    # 2 ---> OPTIMIZER
    translation_registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=10.0,
                                                                             minStep=1e-4,
                                                                             numberOfIterations=500,
                                                                             gradientMagnitudeTolerance=1e-8)

    # 3 ---> INTERPOLATOR
    translation_registration_method.SetInterpolator(Sitk.sitkLinear)

    # 4 ---> TRANSFORMATION
    tx = Sitk.TranslationTransform(fixed_image_itk.GetDimension())
    translation_registration_method.SetInitialTransform(tx)

    # MASK BASED METRIC CALCULATION
    translation_registration_method.SetMetricFixedMask(fixed_mask_itk)
    translation_registration_method.SetMetricMovingMask(moving_mask_itk)

    # Registration execution

    if verbose:
        translation_registration_method.AddCommand(Sitk.sitkIterationEvent, lambda: command_iteration(translation_registration_method))
    calculated_translation_transformation = translation_registration_method.Execute(fixed_image_itk, moving_image_itk)
    print(translation_registration_method.GetOptimizerStopConditionDescription())
    print("")

    print("Translation ended : ", calculated_translation_transformation)

    # Applying the first transformation to the first volume/mask
    moving_mask_itk = Sitk.Resample(moving_mask_itk, fixed_mask_itk, calculated_translation_transformation,
                                    Sitk.sitkNearestNeighbor, 0.0, fixed_image_itk.GetPixelIDValue())
    moving_image_itk = Sitk.Resample(moving_image_itk, fixed_image_itk, calculated_translation_transformation, Sitk.sitkLinear, 0.0,
                                     fixed_image_itk.GetPixelIDValue())

    if is_rotation_needed:
        # --------------------------------------------------------------
        # ------------ SECONDARY  ROTATION/TRANSLATION PART ------------
        # --------------------------------------------------------------
        # Start of registration declaration
        rotation_registration_method = Sitk.ImageRegistrationMethod()

        # 1 ---> METRIC
        # rotation_registration_method.SetMetricAsMeanSquares()
        rotation_registration_method.SetMetricAsCorrelation()

        # 2 ---> OPTIMIZER
        rotation_registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1e-3,
                                                                              minStep=1e-6,
                                                                              numberOfIterations=50,
                                                                              gradientMagnitudeTolerance=1e-5)

        # 3 ---> INTERPOLATOR
        rotation_registration_method.SetInterpolator(Sitk.sitkLinear)

        # 4 ---> TRANSFORMATION
        tx = Sitk.CenteredTransformInitializer(fixed_image_itk,
                                               moving_image_itk,
                                               Sitk.Euler3DTransform(),
                                               Sitk.CenteredTransformInitializerFilter.GEOMETRY)

        rotation_registration_method.SetInitialTransform(tx)

        # MASK BASED METRIC CALCULATION
        rotation_registration_method.SetMetricFixedMask(fixed_mask_itk)
        rotation_registration_method.SetMetricMovingMask(moving_mask_itk)

        # Registration execution
        if verbose:
            rotation_registration_method.AddCommand(Sitk.sitkIterationEvent, lambda: command_iteration(rotation_registration_method)) #Verbose ?
        calculated_rotation_transformation = rotation_registration_method.Execute(fixed_image_itk, moving_image_itk)

        print("Rotation ended")
        # returning both translation and rotation if needed
        return calculated_translation_transformation, calculated_rotation_transformation

    return calculated_translation_transformation


def apply_rotation_pipeline(image, lcl_triangle_angle, rotation_matrix, lcl_throat_coordinates, offset, symmetry_angle):
    """
    applies all the rotations with the previously calculated angles/rotation matrix in order to align the rat with
    the z axis
    :param offset:
    :param image: input image
    :param lcl_triangle_angle: first function angle (in rad)
    :param rotation_matrix: second function rotation matrix
    :param lcl_throat_coordinates: will be used as the center of rotation for the rotation matrix
    :param symmetry_angle: third function angle (in rad)
    :return: rotated image
    """
    image = compute_2d_rotation(image, lcl_triangle_angle, "linear")
    image = compute_3d_rotation(image, rotation_matrix, lcl_throat_coordinates, [0, 0, offset])
    image = compute_2d_rotation(image, symmetry_angle, "linear")

    return image

import os
import PyIPSDK
import segmentation, resampling, ratGistrationIO


if __name__ == "__main__":

    folder_name = "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\R0873_13_AuC_PBS\\Above_Acquisition\\"
    l = folder_name.split("\\")
    ratGistrationIO.remove_last_folder_in_path(folder_name)
