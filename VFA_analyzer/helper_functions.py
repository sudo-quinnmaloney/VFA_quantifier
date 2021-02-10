import cv2
import numpy as np
import math
import copy



def cropImage(cropMe, y_min, y_max, x_min, x_max):
    '''
    This function simply crops the image that we are working with into the specified
    dimensions that we have hard coded: 2540 x 2400 -> width x height
    cropMe[1200:3600, 560:3100]
    :param cropMe: image to be cropped
    :param y_min: lower y bound (actually the top of the image, because Y axis is reversed)
    :param y_max: upper y bound (actually the bottom of the image, because Y axis is reversed)
    :param x_min: lower x bound
    :param x_max: upper x bound
    :return:
     params of return statement are in [Y, X] cropping ranges
    '''

    return cropMe[y_min:y_max, x_min:x_max]



def localizeWithHough(rotatedandscaled, coords, spotnum):
    '''This function corrects any small errors in circle placement  by using a hough transform to detect the actual center of a given spot. Then it updates the pointmap.'''
    image = cropImage(rotatedandscaled, coords[1] - 100, coords[1] + 100, coords[0] - 100, coords[0] + 100)

    #image = cv2.medianBlur(image,5)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    maxIntensity = 255.0 # depends on dtype of image data
    x = np.arange(maxIntensity)

    # Parameters for manipulating image data (phi=.8, theta=5 works quite well for the most recent images)
    phi = .8
    theta = 5

    # Increase intensity such that
    # dark pixels become much brighter,
    # bright pixels become slightly bright
    newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
    newImage0 = np.array(newImage0,dtype=np.uint8)
    
    #cv.HoughCircles (image, circles, method, dp, minDist, param1 = 100, param2 = 100, minRadius, maxRadius)
    circles = cv2.HoughCircles(newImage0, cv2.HOUGH_GRADIENT,4, 20, 1000, 150, minRadius=67, maxRadius=78)
    
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(newImage0,(i[0],i[1]),70,(0,255,0),2)
        # draw the center of the circle
        cv2.circle(newImage0,(i[0],i[1]),2,(0,0,255),3)
        break
    

    cv2.imshow('detected circles',newImage0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('spot'+spotnum+'.jpg',newImage0)

    return (coords[0] + (circles[0][0][0] - 100), coords[1] + (circles[0][0][1] - 100))
    
def localizeWithCentroid(rotatedandscaled, coords, spotnum, display_spots):
    '''This function detects the actual center of a given spot by calculating the centroid of the pixel intensities. Then it updates the pointmap. This is much more effective than the Hough transform approach, but a bit slower.'''
    image = cropImage(rotatedandscaled, coords[1] - 100, coords[1] + 100, coords[0] - 100, coords[0] + 100)

    #image = cv2.medianBlur(image,5)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    avg = np.mean(np.ravel(image))
    ret,thresh = cv2.threshold(image,avg,255,0)
    M = cv2.moments(thresh)
    
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    
    if (display_spots == True):
        maxIntensity = 255.0 # depends on dtype of image data
        x = np.arange(maxIntensity)
        phi = .8
        theta = 5
        newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
        newImage0 = np.array(newImage0,dtype=np.uint8)
        cv2.circle(newImage0,(cX,cY),70,(0,255,0),2)
        cv2.circle(newImage0,(cX,cY),2,(0,0,255),3)
        cv2.imshow('detected circles',newImage0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('spot'+spotnum+'.jpg',newImage0)
        
    return (coords[0] + (cX - 100), coords[1] + (cY - 100))
    
    
def shiftBy(deltaX, deltaY, img):
    '''
    If delta values are negative, the translation matrix will move it the correct direction,
    so we dont have to worry about the negative values
    :param deltaX: shift by this delta x
    :param deltaY: shift by this delta y
    :param img: image to shift
    :return: shifted image
    '''

    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([ [1,0,deltaX], [0,1,deltaY] ])
    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    return img_translation






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
    assert(int(alignA[0]) <= int(alignB[0]))
    deltaX = abs(int(alignA[0]) - int(alignB[0]))
    deltaY = abs(int(alignA[1]) - int(alignB[1]))


    distance = math.sqrt(deltaX * deltaX + deltaY * deltaY)

    ratioToScale = distance_to_compare_with / distance
    #print("Distance: " + str(distance))
    #print("Ratio: " + str(ratioToScale))


    return ratioToScale






def rotateAndScale(img, scaleFactor = 1, degreesCCW = 0):
    '''
    :param img: the image that will get rotated and returned
    :param scaleFactor: option to scale image
    :param degreesCCW: DEGREES NOT RADIANS to rotate CCW. Neg value will turn CW
    :return: rotated image
    '''


    (oldY,oldX) = (img.shape[0], img.shape[1]) #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))

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

    angleToRotate = math.atan(deltaY/deltaX)

    if direction == "CW":
        angleToRotate = -1 * angleToRotate * (180/math.pi)
    else:
        angleToRotate = angleToRotate * (180/math.pi)

    return angleToRotate






def alignImage(image, image_name, correct_distance_from_A_to_B, correct_alignmarker_A_coordinates, template_dictionary):
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
    avg_angle = (angle1 + angle2)/2

    scaleFactor1 = findScaleFactor(alignA, alignB, correct_distance_from_A_to_B)
    scaleFactor2 = findScaleFactor(alignC, alignD, correct_distance_from_A_to_B)
    avg_scale_factor = (scaleFactor1 + scaleFactor2)/2
    #print(avg_scale_factor)


    ############## Basically used as a threshold, given that the test is inserted correctly,
    # it should never be larger than 45 degrees
    if abs(avg_angle) > 45:
        print(image_name + ' is a bad image, it was rotated ' + str(avg_angle) + ' degrees, unexpected amount')


    ########### Actually rotates image
    rotated_image = rotateAndScale(image, avg_scale_factor, avg_angle)




    ############### Shifts the image
    new_alignA = matchTemplate(rotated_image, template_dictionary, "template_A")
    alignAX = new_alignA[0]
    alignAY = new_alignA[1]
    shiftBy_x = correct_alignmarker_A_coordinates[0] - alignAX
    shiftBy_y = correct_alignmarker_A_coordinates[1] - alignAY

    shifted_and_rotated = shiftBy(shiftBy_x, shiftBy_y, rotated_image)


    return shifted_and_rotated






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
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    ######### Partitioning of image
    gray_width, gray_height = gray_image.shape[::-1]
    gray_half_width = gray_width//2 #Note the double '/' for making sure our result is an integer

    if partition == 'B':
        # We are going to look at the right partition
        gray_image = gray_image[0:gray_height,gray_half_width:gray_width]
    elif partition == 'A':
        # We are going to look at the left partition
        gray_image = gray_image[0:gray_height, 0:gray_half_width]




    ########## Actually completing template match
    w,h = template.shape[::-1]
    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)


    ##########This section calculates the midpoint of the square that the template matched to
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    deltaX = bottom_right[0] - top_left[0]
    deltaY = bottom_right[1] - top_left[1]
    midXPoint = top_left[0] + deltaX//2
    midYPoint = top_left[1] + deltaY//2


    #We run this if we were looking at the right side of the image
    #This is because the midpoint for X would be relative to the cropped image we fed into the template matcher
    # However, we have to add 1225 (or whatever half the width of gray_image is) back to get the
    # x value with respect to the entire image
    if partition == 'B':
        midXPoint = midXPoint + gray_half_width

    return (midXPoint, midYPoint)






def create_circular_mask(h, w, center=None, radius=None):
    '''
    **Note: height and width must be exactly the same as the image we are making a mask for
     bc we multiply the matrices together in the end
    :param h: height of the image we are creating a mask for
    :param w: width of the image we are creating a mask for
    :param center: The center point of the circle
    :param radius: The specified radius that we choose: in our case, we are defaulting to 60px
    :return:
    '''

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask






def findAverageLightIntensity(maskedImage, mask):

    '''
    :param maskedImage: This is the original image that has been multiplied with the mask
    :param mask: This is the mask that was made to 'cut out' the
    :return: returns the average, which is found to be the sum of the pixel values divided by the area of the mask
    '''

    sum_of_pixels = np.sum(maskedImage)
    area = np.sum(mask)
    return(sum_of_pixels/area)






def drawCirclesAndLabels(already_aligned_image, pointMap, radius_to_draw):
    '''
    This function is just to display the image with the labels that we predetermined,
    it has no impact on the resulting calculations
    :param image: The image that is going to be drawn on. This is the image that is ALREADY ALIGNED.
    :param pointMap: The pointmap that we have predefined
    :param radius_to_draw: The radius of each circle that will be drawn
    :return:
    '''

    ############This is because of Python's pass by object reference,
    # in order to not modify the version passed in, we make a deep copy
    copyImage = copy.deepcopy(already_aligned_image)


    ##NOTE:
    ##'key' is the name of the circle/alignment marker
    ##'value' is the coordinate of its respective circle/alignment marker
    for key, value in pointMap.items():

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        color = (255, 255, 255)
        thickness = 2

        if key not in ['A','B','C','D']:

            cv2.circle(copyImage, value, radius_to_draw, color, thickness)

        copyImage = cv2.putText(copyImage, key, value, font,
                            fontScale, color, thickness, cv2.LINE_AA)
    return copyImage
 
