import numpy as np

# -- registration library --
import SimpleITK as Sitk

import PyIPSDK.IPSDKIPLBinarization as Bin

import math

# -- 2D Convex Hull function --
import ratGistrationIO
from skimage.measure import label, regionprops


def apply_2d_rotation(vector, angle):
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

    c = math.cos(float(angle))
    s = math.sin(float(angle))

    skull_to_center_vector = np.array([skull_center[0] - image.shape[2]/2,
                                       skull_center[1] - image.shape[1]/2])
    new_skull_to_center_vector = apply_2d_rotation(skull_to_center_vector, angle)

    jaw_one_to_center_vector = np.array([baryctr_jaw_one[0] - image.shape[2] / 2,
                                         baryctr_jaw_one[1] - image.shape[1] / 2])
    new_jaw_one_to_center_vector = apply_2d_rotation(jaw_one_to_center_vector, angle)

    if new_jaw_one_to_center_vector[1] < new_skull_to_center_vector[1]:
        angle += math.pi
        c = -c
        s = -s

    image_itk = Sitk.GetImageFromArray(image)
    skull_itk = Sitk.GetImageFromArray(skull)

    tx = Sitk.AffineTransform(image_itk.GetDimension())
    tx.SetMatrix((c, s, 0,
                  -s, c, 0,
                  0, 0, 1))
    interpolator = Sitk.sitkLinear

    tx.SetCenter((image.shape[2]/2, image.shape[1] / 2, 0))
    image_itk = Sitk.Resample(image_itk, tx, interpolator, 0.0, image_itk.GetPixelIDValue())
    skull_itk = Sitk.Resample(skull_itk, tx, interpolator, 0.0, skull_itk.GetPixelIDValue())

    resulting_skull = Sitk.GetArrayFromImage(skull_itk)
    resulting_image = Sitk.GetArrayFromImage(image_itk)

    return np.copy(resulting_image), np.copy(resulting_skull), float(angle)


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


def straight_throat_rotation(image, throat_mask_img):

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
                vectors_list.append(np.array([centroid_list[-1][0] - centroid_list[-2][0],
                                              centroid_list[-1][1] - centroid_list[-2][1],
                                              z_length]))
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

    #normalized_total_vector_itk = np.copy(normalized_total_vector)
    #normalized_total_vector_itk[1] = normalized_total_vector[0]
    #normalized_total_vector_itk[0] = normalized_total_vector[1]

    print("Normalized total vector: ", normalized_total_vector)
    print("Flipped vector :", normalized_total_vector)

    center_of_rotation = [centroid_list[0][0], centroid_list[0][1]]

    image_itk = Sitk.GetImageFromArray(image)

    c = np.dot(normalized_total_vector, np.array([0, 0, 1]))
    s = np.linalg.norm(np.cross(normalized_total_vector, np.array([0, 0, 1])))
    u = normalized_total_vector[0]
    v = normalized_total_vector[1]
    w = normalized_total_vector[2]

    # Matrix calculation
    matrix = [[u * u * (1 - c) + c, u * v * (1 - c) - w * s, u * w * (1 - c) + v * s],
              [u * v * (1 - c) + w * s, v * v * (1 - c) + c, v * w * (1 - c) - u * s],
              [u * w * (1 - c) - v * s, v * w * (1 - c) + u * s, w * w * (1 - c) + c]]

    tx = Sitk.AffineTransform(image_itk.GetDimension())

    tx.SetMatrix((matrix[0][0], matrix[0][1], matrix[0][2],
                  matrix[1][0], matrix[1][1], matrix[1][2],
                  matrix[2][0], matrix[2][1], matrix[2][2]))

    print("Center :", center_of_rotation[0], center_of_rotation[1], 0)
    tx.SetCenter((0, center_of_rotation[1], center_of_rotation[0]))
    interpolator = Sitk.sitkLinear

    # FINDERREALSEARCH
    tx.SetTranslation((0, 0, 100))

    image_itk = Sitk.Resample(image_itk, tx, interpolator, 0.0, image_itk.GetPixelIDValue())

    image = Sitk.GetArrayFromImage(image_itk)
    return np.copy(image), center_of_rotation, u, v


def symmetry_based_registration(image, skull, skull_bounding_box, throat_coordinates, number_of_iterations):
    """
    Looking for the best rotation (around axis Z) making the skull symmetric
    :param image: input Image
    :param skull: input Image skull mask
    :param skull_bounding_box: skull bounding box (2D)
    :param throat_coordinates: coordinates of the segmented throat
    :param number_of_iterations: how many angles we're trying to check (0.5 degrees step)
    :return: rotated image, calculated angle
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

        angle = -number_of_iterations + i

        c = math.cos(float(angle) / 180 * math.pi)
        s = math.sin(float(angle) / 180 * math.pi)

        image_copy = np.copy(image)
        skull_copy = np.copy(skull)

        image_itk = Sitk.GetImageFromArray(image_copy)
        skull_itk = Sitk.GetImageFromArray(skull_copy)

        tx = Sitk.AffineTransform(image_itk.GetDimension())
        tx.SetMatrix((c, s, 0,
                      -s, c, 0,
                      0, 0, 1))

        tx.SetCenter((throat_coordinates[1], throat_coordinates[0], 0))
        image_itk = Sitk.Resample(image_itk, tx, Sitk.sitkLinear, 0.0, image_itk.GetPixelIDValue())
        skull_itk = Sitk.Resample(skull_itk, tx, Sitk.sitkNearestNeighbor, 0.0, skull_itk.GetPixelIDValue())

        resulting_image = Sitk.GetArrayFromImage(image_itk)
        resulting_skull = Sitk.GetArrayFromImage(skull_itk)

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

    c = math.cos(float(correct_angle) / 180 * math.pi)
    s = math.sin(float(correct_angle) / 180 * math.pi)

    image_itk = Sitk.GetImageFromArray(image_copy)
    skull_itk = Sitk.GetImageFromArray(skull_copy)

    tx = Sitk.AffineTransform(image_itk.GetDimension())
    tx.SetMatrix((c, s, 0,
                  -s, c, 0,
                  0, 0, 1))
    tx.SetCenter((throat_coordinates[1], throat_coordinates[0], 0))
    image_itk = Sitk.Resample(image_itk, tx, Sitk.sitkLinear, 0.0, image_itk.GetPixelIDValue())
    skull_itk = Sitk.Resample(skull_itk, tx, Sitk.sitkNearestNeighbor, 0.0, skull_itk.GetPixelIDValue())
    resulting_image = Sitk.GetArrayFromImage(image_itk)

    cropped_image = resulting_image[:, skull_bounding_box[2]:skull_bounding_box[3] + 1, skull_bounding_box[0]:skull_bounding_box[1] + 1]

    return np.copy(cropped_image), correct_angle


def command_iteration(method):
    """
    registration verbose output function
    :param method: registration method
    :return: None
    """
    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),
                                     method.GetMetricValue()))
    # print("Position = ", method.GetOptimizerPosition())


def registration_computation_with_mask(moving_image, reference_image, moving_mask, reference_mask, is_rotation_needed = False, verbose = False):
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


import PyIPSDK
import segmentation, resampling


if __name__ == "__main__":

    folder_name = "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\R0873_13_AuC_PBS\\Above_Acquisition\\"
    list_of_files = ratGistrationIO.create_list_of_files(folder_name, 'tif')
    image_test = resampling.bin_resize(ratGistrationIO.open_seq(list_of_files), 2)

    img_ipsdk = PyIPSDK.fromArray(image_test)

    threshold_value = segmentation.find_threshold_value("Au")
    print("Threshold Value :", threshold_value)

    # -- Threshold computation
    thresholded_img_ipsdk = Bin.thresholdImg(img_ipsdk, threshold_value, 3)

    # -- Extracting skull
    above_skull, skull_bbox, \
    barycenter_jaw_one, barycenter_jaw_two, \
    y_max_jaw_one, y_max_jaw_two = segmentation.extract_skull_and_jaws(thresholded_img_ipsdk)
    ratGistrationIO.save_tif_sequence(above_skull, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\R0873_13_AuC_PBS\\Skull_result\\")

    straightened_image, above_skull, triangle_angle = straight_triangle_rotation(np.copy(image_test), above_skull,
                                                                                 skull_bbox, barycenter_jaw_one,
                                                                                 barycenter_jaw_two)
    ratGistrationIO.save_tif_sequence(straightened_image, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\R0873_13_AuC_PBS\\Straightened_image\\")

    above_skull_ipsdk = PyIPSDK.fromArray(above_skull)
    bbox = segmentation.skull_bounding_box_retriever(above_skull_ipsdk)

    ##################### + WRITING VOLUME PART + #####################

    # Cropping the volumes and the skull masks
    throat_mask = segmentation.throat_segmentation(straightened_image, bbox, "Au")

    ratGistrationIO.save_tif_sequence(throat_mask, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\R0873_13_AuC_PBS\\Throat_mask\\")
    # FORMANUAL1
    straightened_image, throat_coordinates, u, v = straight_throat_rotation(straightened_image, throat_mask)

    ratGistrationIO.save_tif_sequence(straightened_image, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\R0873_13_AuC_PBS\\Straight_final_image\\")