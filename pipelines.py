import PyIPSDK
import PyIPSDK.IPSDKIPLBinarization as Bin

import numpy as np

import segmentation, resampling, registration, ratGistrationIO


def bothSkullExtractionPipeline(input_above_folder, input_below_folder, element="Au"):
    """
    computes all the skull segmentation and rat aligning with z axis calculations
    :param input_above_folder: input above energy acquisition images
    :param input_below_folder: input below energy acquisition images
    :param element: k-edge element (Au, I, Gd...)
    :return: None
    """

    above_list_of_files = ratGistrationIO.create_list_of_files(input_above_folder, 'tif')
    above_image = ratGistrationIO.open_seq(above_list_of_files)
    #below_list_of_files = ratGistrationIO.create_list_of_files(input_below_folder, 'tif')
    #below_image = ratGistrationIO.open_seq(below_list_of_files)

    binning_factor = 2
    binned_image = resampling.bin_resize(above_image, binning_factor)

    img_ipsdk = PyIPSDK.fromArray(binned_image)

    threshold_value = segmentation.find_threshold_value(element)
    print("Threshold Value :", threshold_value)

    # -- Threshold computation
    thresholded_img_ipsdk = Bin.thresholdImg(img_ipsdk, threshold_value, 3)

    # -- Extracting skull
    above_skull, skull_bbox, \
        barycenter_jaw_one, barycenter_jaw_two, \
        y_max_jaw_one, y_max_jaw_two = segmentation.extract_skull_and_jaws(thresholded_img_ipsdk)

    # 1) First rotation based on the position of skull/jaws
    straightened_image, above_skull, triangle_angle = registration.straight_triangle_rotation(np.copy(binned_image),
                                                                                              above_skull,
                                                                                              skull_bbox,
                                                                                              barycenter_jaw_one,
                                                                                              barycenter_jaw_two)

    above_skull_ipsdk = PyIPSDK.fromArray(above_skull)
    bbox = segmentation.skull_bounding_box_retriever(above_skull_ipsdk)

    # Cropping the volumes and the skull masks
    throat_mask = segmentation.throat_segmentation(straightened_image, bbox, element)

    # 2) Second rotation based on the position of the throat
    straightened_image, rotation_matrix, throat_coordinates, offset = registration.straight_throat_rotation(straightened_image,
                                                                                                            throat_mask)

    # We re-segment the skull/jaws
    thresholded_img_ipsdk = Bin.thresholdImg(PyIPSDK.fromArray(straightened_image), threshold_value, 3)

    above_skull, skull_bbox, \
        barycenter_jaw_one, barycenter_jaw_two, \
        y_max_jaw_one, y_max_jaw_two = segmentation.extract_skull_and_jaws(thresholded_img_ipsdk)

    # 3) Third rotation based on the symmetry of the skull
    final_image, final_skull, symmetry_angle = registration.symmetry_based_registration(straightened_image,
                                                                                        above_skull,
                                                                                        skull_bbox,
                                                                                        throat_coordinates,
                                                                                        20)

    # -- Apply the rotations to the original images
    final_above_image = registration.apply_rotation_pipeline(above_image, triangle_angle, rotation_matrix,
                                                             throat_coordinates * binning_factor,
                                                             offset * binning_factor,
                                                             symmetry_angle)

    final_above_image = final_above_image.astype(np.float32)
    thresholded_final_above_image_ipsdk = Bin.thresholdImg(PyIPSDK.fromArray(final_above_image), threshold_value, 3)
    final_above_skull, final_above_skull_bbox = segmentation.extract_skull(thresholded_final_above_image_ipsdk)

    #final_below_image = registration.apply_rotation_pipeline(below_image, triangle_angle, rotation_matrix,
    #                                                         throat_coordinates * binning_factor,
    #                                                         offset * binning_factor,
    #                                                         symmetry_angle)

    #final_below_image = final_below_image.astype(np.float32)
    #thresholded_final_below_image_ipsdk = Bin.thresholdImg(PyIPSDK.fromArray(final_below_image), threshold_value, 3)
    #final_below_skull, final_below_skull_bbox = segmentation.extract_skull(thresholded_final_below_image_ipsdk)

    output_folder = ratGistrationIO.remove_last_folder_in_path(input_above_folder)

    ratGistrationIO.save_tif_sequence_and_crop(final_above_image, final_above_skull_bbox,
                                               output_folder + "Above_img_for_registration\\")
    ratGistrationIO.save_tif_sequence_and_crop(final_above_skull, final_above_skull_bbox,
                                               output_folder + "Above_skull_for_registration\\")

    #ratGistrationIO.save_tif_sequence_and_crop(final_below_image, final_below_skull_bbox,
    #                                           output_folder + "Below_img_for_registration\\")
    #ratGistrationIO.save_tif_sequence_and_crop(final_below_skull, final_below_skull_bbox,
    #                                           output_folder + "Below_skull_for_registration\\")
