import PyIPSDK

import numpy as np

import segmentation, registration


def bothSkullExtractionPipeline(input_above_folder, output_above_img_folder, output_above_skull_folder,
                                input_below_folder, output_below_img_folder, output_below_skull_folder,
                                element="Au"):

    ######################### + LOADING/RESAMPLING PART + #########################
    image = Image3D.Image3D(folderName=input_above_folder)
    image.createListOfFiles('tif')
    image.loadSlices()
    image.imresize(0.5)
    ######################### - LOADING/RESAMPLING PART - #########################

    ########################## + SKULL EXTRACTING PART + ##########################

    # -- Calculating the threshold
    print("Image type:", type(image.data))
    print("image data type:", image.data.dtype, "and shape :", image.data.shape)
    # Initialize the numpy array into an IPSDK image
    img_ipsdk = PyIPSDK.fromArray(image.data)
    threshold_value = segmentation.find_threshold_value(element)
    print("Threshold Value :", threshold_value)

    # -- Threshold computation
    thresholded_img_ipsdk = bin.thresholdImg(img_ipsdk, threshold_value, 3)

    # -- Extracting skull
    above_skull, skull_bbox, \
    barycenter_jaw_one, barycenter_jaw_two, \
    y_max_jaw_one, y_max_jaw_two = segmentation.extract_skull_and_jaws(thresholded_img_ipsdk)

    straightened_image, above_skull, triangle_angle = registration.straight_triangle_rotation(np.copy(image.data),
                                                                                              above_skull, skull_bbox,
                                                                                              barycenter_jaw_one,
                                                                                              barycenter_jaw_two)

    above_skull_ipsdk = PyIPSDK.fromArray(above_skull)
    bbox = segmentation.skull_bounding_box_retriever(above_skull_ipsdk)

    ##################### + WRITING VOLUME PART + #####################

    # Cropping the volumes and the skull masks
    throat_mask = throat_segmentation(straightened_image, bbox, element)

    # FORMANUAL1
    finalRotatedImage, throatCoordinates, secondAngle, Ux, Uy = reOrientThroatWise(rotatedImage, throat_mask)

    saveTif(finalRotatedImage, output_above_img_folder)
    z = input("")
    print(throatCoordinates)

    # FORMANUAL3
    # throatCoordinates = [230, 280]

    finalRotatedIPSDKImage = PyIPSDK.fromArray(finalRotatedImage)
    thresholded_img_ipsdk = bin.thresholdImg(finalRotatedIPSDKImage, threshold_value, 3)

    finalExtractedAboveSkull, skull_bbox, barycenter_jaw_one, BCJTwo, y_max_jaw_one, y_max_jaw_two = extractSkullAndJaws(
        thresholded_img_ipsdk, True)

    if y_max_jaw_one[1] > y_max_jaw_two[1]:
        topJawsOne = [y_max_jaw_one[1], y_max_jaw_one[2]]
        topJawsTwo = [y_max_jaw_two[0], y_max_jaw_two[2]]
    else:
        topJawsOne = [y_max_jaw_one[0], y_max_jaw_one[2]]
        topJawsTwo = [y_max_jaw_two[1], y_max_jaw_two[2]]

    print("throat :", throatCoordinates)

    saveTif(finalRotatedImage, output_above_img_folder)
    #    z = input("")
    finalRotatedImage, finalAngle = symetryAnalysis(finalRotatedImage, finalExtractedAboveSkull, skull_bbox,
                                                    throatCoordinates)
    newFinalRotatedImage = np.copy(finalRotatedImage)
    newFinalRotatedImage = newFinalRotatedImage.astype(np.float32)
    newFinalRotatedIPSDKImage = PyIPSDK.fromArray(newFinalRotatedImage)
    newThresholdedIPSDKImage = bin.lightThresholdImg(newFinalRotatedIPSDKImage, threshold_value)

    finalExtractedAboveSkull, skull_bbox, barycenter_jaw_one, BCJTwo, y_max_jaw_one, y_max_jaw_two = extractSkullAndJaws(
        newThresholdedIPSDKImage, True)

    aboveImage = Image3D.Image3D(folderName=input_above_folder)
    aboveImage.createListOfFiles('tif')
    aboveImage.loadSlices()

    aboveImage = applyAngles(aboveImage.data, triangle_angle, throatCoordinates, secondAngle, Ux, Uy, finalAngle)

    newFinalRotatedImage = aboveImage.astype(np.float32)
    newFinalRotatedIPSDKImage = PyIPSDK.fromArray(newFinalRotatedImage)
    newThresholdedIPSDKImage = bin.lightThresholdImg(newFinalRotatedIPSDKImage, threshold_value)
    finalExtractedAboveSkull, skull_bbox, barycenter_jaw_one, BCJTwo, y_max_jaw_one, y_max_jaw_two = extractSkullAndJaws(
        newThresholdedIPSDKImage, True)
    print("Retrieve above BB :", skull_bbox)
    print("Gauche :", 2 * throatCoordinates[1] - skull_bbox[0], "Droite :", skull_bbox[1] - 2 * throatCoordinates[1])
    if 2 * throatCoordinates[0] - skull_bbox[0] > skull_bbox[1] - 2 * throatCoordinates[0]:
        skull_bbox[1] = int(skull_bbox[0] + (2 * throatCoordinates[0] - skull_bbox[0]) * 2)
    else:
        skull_bbox[0] = int(skull_bbox[1] - (skull_bbox[1] - 2 * throatCoordinates[0]) * 2)
    print("Above BB :", skull_bbox)
    croppedImage = newFinalRotatedImage[:, skull_bbox[2]:skull_bbox[3] + 1, skull_bbox[0]:skull_bbox[1] + 1]
    croppedSkull = finalExtractedAboveSkull[:, skull_bbox[2]:skull_bbox[3] + 1, skull_bbox[0]:skull_bbox[1] + 1]

    print(croppedImage.shape)
    saveTif(croppedImage, output_above_img_folder)
    saveTif(croppedSkull, output_above_skull_folder)

    # FORMANUAL4
    # throatCoordinates = [208, 258]
    belowImage = Image3D.Image3D(folderName=input_below_folder)
    belowImage.createListOfFiles('tif')
    belowImage.loadSlices()
    belowImage = applyAngles(belowImage.data, triangle_angle, throatCoordinates, secondAngle, Ux, Uy, finalAngle)

    newFinalRotatedImage = belowImage.astype(np.float32)
    newFinalRotatedIPSDKImage = PyIPSDK.fromArray(newFinalRotatedImage)
    newThresholdedIPSDKImage = bin.lightThresholdImg(newFinalRotatedIPSDKImage, threshold_value)
    finalExtractedAboveSkull, skull_bbox, barycenter_jaw_one, BCJTwo, y_max_jaw_one, y_max_jaw_two = extractSkullAndJaws(
        newThresholdedIPSDKImage, True)
    print("Retrieve below BB :", skull_bbox)
    if 2 * throatCoordinates[1] - skull_bbox[0] > skull_bbox[1] - 2 * throatCoordinates[1]:
        skull_bbox[1] = int(skull_bbox[0] + (2 * throatCoordinates[1] - skull_bbox[0]) * 2)
    else:
        skull_bbox[0] = int(skull_bbox[1] - (skull_bbox[1] - 2 * throatCoordinates[1]) * 2)

    print("Below BB :", skull_bbox)
    croppedImage = newFinalRotatedImage[:, skull_bbox[2]:skull_bbox[3] + 1, skull_bbox[0]:skull_bbox[1] + 1]
    croppedSkull = finalExtractedAboveSkull[:, skull_bbox[2]:skull_bbox[3] + 1, skull_bbox[0]:skull_bbox[1] + 1]

    saveTif(croppedImage, output_below_img_folder)
    saveTif(croppedSkull, output_below_skull_folder)
    croppedSkull[croppedSkull == 1] = 255