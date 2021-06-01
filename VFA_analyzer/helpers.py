import cv2
import numpy as np
import math

MARGIN = 90
MASK = np.zeros((1, 1))


def localizeWithCentroid(rotatedandscaled, coord_dict, display_spots, error):
    '''This function detects the actual center of a given spot by calculating the centroid of the pixel intensities. Then it updates the pointmap. This is much more effective than the Hough transform approach, but a bit slower.'''
    checkLocalization = False
    output_dict = {}
    for i in coord_dict.keys():
        try:
            coords = coord_dict[i]

            if i not in ['A', 'B', 'C', 'D']:
                cropped = rotatedandscaled[coords[1] - MARGIN:coords[1] + MARGIN, coords[0] - MARGIN: coords[0] + MARGIN]

                # image = cv2.medianBlur(image,5)
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                avg = np.mean(np.ravel(cropped))
                ret, thresh = cv2.threshold(cropped, avg, 255, 0)
                M = cv2.moments(thresh)

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if (display_spots == True):
                    maxIntensity = 255.0  # depends on dtype of image data
                    x = np.arange(maxIntensity)
                    phi = .8
                    theta = 5
                    newImage0 = (maxIntensity / phi) * (cropped / (maxIntensity / theta)) ** 0.5
                    newImage0 = np.array(newImage0, dtype=np.uint8)
                    cv2.circle(newImage0, (cX, cY), 70, (0, 255, 0), 2)
                    cv2.circle(newImage0, (cX, cY), 2, (0, 0, 255), 3)
                    cv2.imshow('detected circles', newImage0)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    cv2.imwrite('spot{}.jpg'.format(i), newImage0)

                output_dict[i] = (coords[0] + (cX - MARGIN), coords[1] + (cY - MARGIN))
            else:
                output_dict[i] = (coords[0], coords[1])

        except:
            checkLocalization = True
            continue
    if (checkLocalization):
        print('\t\t^^Check localization^^')
    return output_dict, (checkLocalization | error)


def findScaleFactor(alignA, alignB, distance_to_compare_with):
    '''
    The purpose of this function is to find the scaling that we should apply to the image during the alignment process.
    We have a set value (based off of our map) that should be the distance between alignment markers A & B.
    This is what distance_to_compare_with represents.
    Then, using the coordinates from the template match, we calculate what the actual distance between the two alignment markers are
    and compute a ratio which will signal what we should scale the image by to get it to match our map.
    NOTE: This function uses distance_to_compare_with, which is a HORIZONTAL relationship.
    i.e. A & B have the same Y-value (horizontal value).
    If you tried to do this for vertical alignment markers, e.g. A & C, the value would be wrong
    :param alignA: The alignment marker on the left
    :param alignB: The alignment marker on the right
    :param distance_to_compare_with: What the distance between them SHOULD be (to create a ratio)
    :return:
    '''
    assert (int(alignA[0]) <= int(alignB[0]))
    deltaX = abs(int(alignA[0]) - int(alignB[0]))
    deltaY = abs(int(alignA[1]) - int(alignB[1]))

    distance = math.sqrt(deltaX * deltaX + deltaY * deltaY)

    ratioToScale = distance_to_compare_with / distance
    # print("Distance: " + str(distance))
    # print("Ratio: " + str(ratioToScale))

    return ratioToScale


def rotateAndScale(img, scaleFactor=1, degreesCCW=0):
    '''
    :param img: the image that will get rotated and returned
    :param scaleFactor: option to scale image
    :param degreesCCW: DEGREES NOT RADIANS to rotate CCW. Neg value will turn CW
    :return: rotated image
    '''

    (oldY, oldX) = (img.shape[0], img.shape[1])  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degreesCCW,
                                scale=scaleFactor)  # rotate about center of image.

    # choose a new image size.
    newX, newY = oldX * scaleFactor, oldY * scaleFactor
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))

    return rotatedImg


def findAngle(alignA, alignB):
    '''
    This function finds the corresponding angle between the two alignment markers passed in, and also returns which
     direction they should be turned to be on the same axis.
     Also note that the order in which these are passed in is also important.
     We always pass them in from left to right when we look at the image,
     e.g. always alignA, alignB or alignC, alignD
        but never alignB, alignA
    -----> RETURNS ANGLE IN DEGREES <-----
    :param alignA: alignment marker A
    :param alignB: alignment marker B
    :return: angleINDEGREES
    '''

    deltaX = abs(int(alignA[0]) - int(alignB[0]))
    deltaY = int(alignA[1]) - int(alignB[1])

    if (deltaY) >= 0:
        direction = "CW"
    else:
        direction = "CCW"

    deltaY = abs(deltaY)

    angleToRotate = math.atan(deltaY / deltaX)

    if direction == "CW":
        angleToRotate = -1 * angleToRotate * (180 / math.pi)
    else:
        angleToRotate = angleToRotate * (180 / math.pi)

    return angleToRotate


def alignImage(image, image_name, correct_distance_from_A_to_B, correct_alignmarker_A_coordinates, template_dictionary, error = False):
    '''
    This function combines the shiftBy function and the rotateImage function into one.
    Essentially places our image on our predetermined grid. First it rotates the image (using avg angle between A-B & C-D),
    and then it finds the new coordinates for Alignment Marker A. Using these new coordinates,
    it shifts the entire image such that the new Alignment Marker A coordinates are in the spot we want them to be.
    FOR THIS SETUP, WE WANT ALIGNMENT MARKER A TO BE ON correct COORDINATES
    For example, in tiff image version, we want alignment marker A at: -> (591, 528)
    This value is a constant up top named ALIGNMENT_MARKER_A_MAP_LOCATION^
    :param image: image to be aligned
    :param image_name: this is the name of the image, for example: an example image name might be -> myimage.tif
        We use this just to print out which image was bad if a image is rotated more than
        the threshold specified below(typically 45 degrees)
    :param correct_distance_from_A_to_B: This is what the distance SHOULD be between A & B.
        We use this to find the scaling factor in order to improve our accuracy and reliability.
    :param correct_alignmarker_A_coordinates: This is where Alignment marker A SHOULD be.
        We use this in order to know how much to shift the image by
        THis should be in the format of (x, y)
    :param template_dictionary: This is the dictionary that basically translates which alignment marker we are
     trying to match with its file name in our local repository.
     An example template dictionary might be as shown below:
     template_dictionary = {
        'template_A': 'alignment_A.tif',
        'template_B': 'alignment_B.tif',
        'template_C': 'alignment_C.tif',
        'template_D': 'alignment_D.tif'
    }
    :return: shifted and rotated image
    '''

    ############## Preparing to rotate and scale the image
    alignA = matchTemplate(image, template_dictionary, "template_A")
    alignB = matchTemplate(image, template_dictionary, "template_B")
    alignC = matchTemplate(image, template_dictionary, "template_C")
    alignD = matchTemplate(image, template_dictionary, "template_D")

    angle1 = findAngle(alignA, alignB)
    angle2 = findAngle(alignC, alignD)
    if abs(angle1) > 30 or abs(angle2) > 30:
        print('\t--Difficult alignment for {}--'.format(image_name))
        error = True
        fallback = np.argmin([abs(angle1), abs(angle2)])
        avg_angle = [angle1, angle2][fallback]
        avg_scale_factor = findScaleFactor(alignA, alignB, correct_distance_from_A_to_B) if not fallback \
            else findScaleFactor(alignC, alignD, correct_distance_from_A_to_B)
    else:
        avg_angle = (angle1 + angle2) / 2
        scaleFactor1 = findScaleFactor(alignA, alignB, correct_distance_from_A_to_B)
        scaleFactor2 = findScaleFactor(alignC, alignD, correct_distance_from_A_to_B)
        avg_scale_factor = (scaleFactor1 + scaleFactor2) / 2
        # print(avg_scale_factor)

    ########### Actually rotates image
    image = rotateAndScale(image, avg_scale_factor, avg_angle)

    ############### Shifts the image
    new_alignA = matchTemplate(image, template_dictionary, "template_A")
    alignAX = new_alignA[0]
    alignAY = new_alignA[1]
    shiftBy_x = correct_alignmarker_A_coordinates[0] - alignAX
    shiftBy_y = correct_alignmarker_A_coordinates[1] - alignAY

    num_rows, num_cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, shiftBy_x], [0, 1, shiftBy_y]])
    image = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))

    return image, error


def matchTemplate(image, template_dictionary, template):
    '''
    This function finds the alignment markers which our program uses to correctly orient the image to our grid
    If we changed our template to another value(added new template, modified old one, etc), we would simply add the
    option to our dictionary and then add the image to our alignment_templates directory.
    This function partitions the image into two sections: the right side and the left side
    This is in order to not confuse the templates (mainly due to the fact that alignment marker B & C are identical)
    :param image: image to match template to
    :param template: input option to determine the template to use
    :return:
    '''

    if template == 'template_A' or template == 'template_C':
        partition = 'A'
    else:
        partition = 'B'

    # Reads the template image from the alignment_templates directory
    template = cv2.imread('alignment templates/' + template_dictionary[template], cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ######### Partitioning of image
    gray_width, gray_height = image.shape[::-1]
    gray_half_width = gray_width // 2  # Note the double '/' for making sure our result is an integer

    if partition == 'B':
        # We are going to look at the right partition
        image = image[0:gray_height, gray_half_width:gray_width]
    elif partition == 'A':
        # We are going to look at the left partition
        image = image[0:gray_height, 0:gray_half_width]

    ########## Actually completing template match
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    ##########This section calculates the midpoint of the square that the template matched to
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    deltaX = bottom_right[0] - top_left[0]
    deltaY = bottom_right[1] - top_left[1]
    midXPoint = top_left[0] + deltaX // 2
    midYPoint = top_left[1] + deltaY // 2

    # We run this if we were looking at the right side of the image
    # This is because the midpoint for X would be relative to the cropped image we fed into the template matcher
    # However, we have to add 1225 (or whatever half the width of gray_image is) back to get the
    # x value with respect to the entire image
    if partition == 'B':
        midXPoint = midXPoint + gray_half_width

    return (midXPoint, midYPoint)

def generateMask(r):
    global MASK
    x, y = r, r
    MASK = np.ones((2*r,2*r))
    for xs in range(r):
        for ys in range(r):
            if (0 < xs ** 2 + ys ** 2 < r ** 2):
                MASK[y + ys][x + xs] = 0
                MASK[y - ys][x - xs] = 0
                MASK[y + ys][x - xs] = 0
                MASK[y - ys][x + xs] = 0
    return MASK

def getStats(image, r, center, commands, visualize, threshold):
    x, y = center[0], center[1]
    thisSpot = np.array(image)[y-r:y+r, x-r:x+r]
    thisSpot = np.ma.array(thisSpot, mask=MASK)
    ravelled = thisSpot.compressed()
    refStd, refMean, refMax = np.std(ravelled), np.mean(ravelled), max(ravelled)
    thisSpot = np.ma.masked_outside(thisSpot, refMean - threshold[0] * refStd, refMean + threshold[0] * refStd)
    refMax = max(thisSpot.compressed())
    thisSpot = np.ma.masked_greater(thisSpot, threshold[1] * refMax)
    thisSpot = np.ma.masked_less(thisSpot, threshold[2] * refMax)
    if (visualize):
        to_show = np.concatenate((thisSpot.filled(255), image[y-r:y+r, x-r:x+r]), axis=1)
        cv2.imshow('spot mask vs original', to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    thisSpot = thisSpot.compressed()
    options = [np.std, np.mean, np.amax, np.amin]
    result = [options[c](thisSpot) for c in range(len(commands)) if commands[c] != 0]
    return result


def drawCirclesAndLabels(image, pointMap, radius_to_draw):
    '''
    This function is just to display the image with the labels that we predetermined,
    it has no impact on the resulting calculations
    :param image: The image that is going to be drawn on. This is the image that is ALREADY ALIGNED.
    :param pointMap: The pointmap that we have predefined
    :param radius_to_draw: The radius of each circle that will be drawn
    :return:
    '''

    for key, value in pointMap.items():

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        color = (255, 255, 255)
        thickness = 2

        if key not in ['A', 'B', 'C', 'D']:
            cv2.circle(image, value, radius_to_draw, color, thickness)

        cv2.putText(image, key, value, font,
                                fontScale, color, thickness, cv2.LINE_AA)
    return image
