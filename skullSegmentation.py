# -- IPSDK Library --
import PyIPSDK
import PyIPSDK.IPSDKIPLMorphology as Morpho
import PyIPSDK.IPSDKIPLAdvancedMorphology as AdvMorpho
import PyIPSDK.IPSDKIPLShapeSegmentation as ShapeSegmentation
import PyIPSDK.IPSDKIPLShapeAnalysis as ShapeAnalysis
import PyIPSDK.IPSDKIPLArithmetic as Arithm

# -- numpy --
import numpy as np


def find_threshold_value(energy_element):
    """
    We return the energy corresponding attenuation value used for bone segmentation
    :param energy_element: what k-edge element are we trying to quantify (Au, I, Gd..)
    :return:
    """
    if energy_element == "Au":
        return 0.26
    else:
        return 0.65

    return 0


def extract_skull(thresholded_ipsdk_image):
    """
    extracting the skull from the volume
    :param thresholded_ipsdk_image: input image
    :return: a mask with the skull only, its bounding box [minX, maxX, minY, maxY, minZ, maxZ]
    """
    # We start with a 3d opening image computation
    morpho_mask = PyIPSDK.sphericalSEXYZInfo(0)  # 3D sphere (r=1) structuring element
    opened_image = Morpho.opening3dImg(thresholded_ipsdk_image, morpho_mask)

    # We extract the biggest shape from the volume (the skull)
    extracted_skull = AdvMorpho.keepBigShape3dImg(opened_image, 1)

    # We'll now analyze its shape by labeling it
    in_label_img_3d = AdvMorpho.connectedComponent3dImg(extracted_skull)

    # and then making it a shape object
    in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(in_label_img_3d)

    # these measures will help us cropping the full volume, getting rid of non-interesting parts of the imaged rat/mice
    # definition of proceeded measure
    in_measure_info_set_3d = PyIPSDK.createMeasureInfoSet3d()
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinXMsrInfo", "BoundingBoxMinXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxXMsrInfo", "BoundingBoxMaxXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinYMsrInfo", "BoundingBoxMinYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxYMsrInfo", "BoundingBoxMaxYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinZMsrInfo", "BoundingBoxMinZMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxZMsrInfo", "BoundingBoxMaxZMsr")

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_ipsdk_image, in_shape_3d_coll, in_measure_info_set_3d)

    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")
    out_bounding_box_max_y_msr_info = out_measure_set.getMeasure("BoundingBoxMaxYMsrInfo")
    out_bounding_box_min_z_msr_info = out_measure_set.getMeasure("BoundingBoxMinZMsrInfo")
    out_bounding_box_max_z_msr_info = out_measure_set.getMeasure("BoundingBoxMaxZMsrInfo")

    # retrieving the extracted skull bounding box
    bounding_box = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_min_z_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_z_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    # we return the mask of the skull and its 3D bounding box
    return extracted_skull, bounding_box


def extract_skull_and_jaws(thresholded_ipsdk_image):
    """
    extracting the skull from the volume
    :param thresholded_ipsdk_image: input volume
    :return: skull mask, skull bounding box (X and Y), jaw one and jaw two barycenter, jaw one and jaw two max Y value
    """

    # We start with a 3d opening image computation

    # We extract the biggest shape from the volume (the skull)
    extracted_skull = AdvMorpho.keepBigShape3dImg(thresholded_ipsdk_image, 1)

    remaining_objects = Arithm.subtractImgImg(thresholded_ipsdk_image, extracted_skull)
    remaining_objects = bin.lightThresholdImg(remaining_objects, 1)
    jaw_one = AdvMorpho.keepBigShape3dImg(remaining_objects, 1)

    remaining_objects = Arithm.subtractImgImg(remaining_objects, jaw_one)
    remaining_objects = bin.lightThresholdImg(remaining_objects, 1)
    jaw_two = AdvMorpho.keepBigShape3dImg(remaining_objects, 1)

    in_measure_info_set_3d = PyIPSDK.createMeasureInfoSet3d()
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinXMsrInfo", "BoundingBoxMinXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxXMsrInfo", "BoundingBoxMaxXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinYMsrInfo", "BoundingBoxMinYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxYMsrInfo", "BoundingBoxMaxYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BarycenterXMsrInfo", "BarycenterXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BarycenterYMsrInfo", "BarycenterYMsr")

    # Skull
    # We'll now analyze its shape by labeling it
    extracted_skull_in_label_img_3d = AdvMorpho.connectedComponent3dImg(extracted_skull)

    # and then making it a shape object
    extracted_skull_in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(extracted_skull_in_label_img_3d)

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_ipsdk_image, extracted_skull_in_shape_3d_coll,
                                                    in_measure_info_set_3d)

    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")
    out_bounding_box_max_y_msr_info = out_measure_set.getMeasure("BoundingBoxMaxYMsrInfo")

    # retrieving the extracted skull bounding box
    skull_bounding_box = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_bounding_box_max_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    # Jaw one
    # We'll now analyze its shape by labeling it
    jaw_one_in_label_img_3d = AdvMorpho.connectedComponent3dImg(jaw_one)

    # and then making it a shape object
    jaw_one_in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(jaw_one_in_label_img_3d)

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_ipsdk_image, jaw_one_in_shape_3d_coll, in_measure_info_set_3d)

    out_barycenter_x_msr_info = out_measure_set.getMeasure("BarycenterXMsrInfo")
    out_barycenter_y_msr_info = out_measure_set.getMeasure("BarycenterYMsrInfo")
    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")

    # retrieving the extracted skull bounding box
    jaw_one_barycenter = [int(out_barycenter_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_barycenter_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    jaw_one_max_x_y = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                       int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                       int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    # Jaw Two
    # We'll now analyze its shape by labeling it
    jaw_two_in_label_img_3d = AdvMorpho.connectedComponent3dImg(jaw_two)

    # and then making it a shape object
    jaw_two_in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(jaw_two_in_label_img_3d)

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_ipsdk_image, jaw_two_in_shape_3d_coll, in_measure_info_set_3d)

    out_barycenter_x_msr_info = out_measure_set.getMeasure("BarycenterXMsrInfo")
    out_barycenter_y_msr_info = out_measure_set.getMeasure("BarycenterYMsrInfo")
    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")

    # retrieving the extracted skull bounding box
    jaw_two_barycenter = [int(out_barycenter_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_barycenter_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    jaw_two_max_x_y = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                       int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                       int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    return np.copy(extracted_skull.array), skull_bounding_box, jaw_one_barycenter, jaw_two_barycenter, jaw_one_max_x_y, jaw_two_max_x_y


def skull_bounding_box_retriever(skull_image):
    """
    Calculating the skull's bounding box (3D)
    :param skull_image: input skull image
    :return: skull's bounding box ([minX, maxX, minY, maxY, minZ, maxZ])
    """
    skull_image = bin.lightThresholdImg(skull_image, 1)

    # We'll now analyze its shape by labeling it
    in_label_img_3d = AdvMorpho.connectedComponent3dImg(skull_image)

    # and then making it a shape object
    in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(in_label_img_3d)

    # these measures will help us cropping the full volume, getting rid of non-interesting parts of the imaged rat/mice
    # definition of proceeded measure
    in_measure_info_set_3d = PyIPSDK.createMeasureInfoSet3d()
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinXMsrInfo", "BoundingBoxMinXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxXMsrInfo", "BoundingBoxMaxXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinYMsrInfo", "BoundingBoxMinYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxYMsrInfo", "BoundingBoxMaxYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinZMsrInfo", "BoundingBoxMinZMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxZMsrInfo", "BoundingBoxMaxZMsr")

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(skull_image, in_shape_3d_coll, in_measure_info_set_3d)

    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")
    out_bounding_box_max_y_msr_info = out_measure_set.getMeasure("BoundingBoxMaxYMsrInfo")
    out_bounding_box_min_z_msr_info = out_measure_set.getMeasure("BoundingBoxMinZMsrInfo")
    out_bounding_box_max_z_msr_info = out_measure_set.getMeasure("BoundingBoxMaxZMsrInfo")

    # retrieving the extracted skull bounding box
    bounding_box = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_min_z_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_z_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    return bounding_box
