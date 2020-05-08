'''
HOW THIS PROGRAM WORKS ON A HIGH LEVEL:


1. First it crops the image to our specified dimensions, to which right now it is set to 1670x1600 -> width x height

2. Then it uses template matching in order to find the top two alignment markers, and using the points
of the alignment markers, it rotates the image.

3. Then it shifts the image to where the first alignment marker, Alignment Marker A, is at the spot we
need it to be now that the image is upright and oriented correctly. In our case, we want that alignment
marker A to be at the point (296, 291). Once the image is aligned, we then know where the other points are
because of the predefined grid that we made.


4. Once the image is aligned, we make a mask for each individual circle, and multiply (element-wise) it by the original
image to create a new image that outside of the mask, it is completely black (matrix value of 0) We calculate
the average light intensity by taking the sum of the image value (inside the mask, the values will remain
their original values) divided by the sum of the mask
(this is essentially just the area of the circle because inside the mask,
there are only 1's and outside it there are only 0's)

We repeat this process for every image in the fluorescent/ directory, which houses all our images that need analysis

'''


import cv2
import numpy as np
import math
import copy
import csv

#This is for reading the images that are in the fluorescent/ directory
from os import listdir
from os.path import isfile, join





def cropImage(cropMe):
    '''
    This function simply crops the image that we are working with into the specified
    dimensions that we have hard coded: 1670 x 1600 -> width x height
    :param cropMe: image to be cropped
    :return:
     params of return statement are in [Y, X] cropping ranges
    '''

    return cropMe[1200:3600, 560:3100]






def matchTemplate(image, template):
    '''
    This function finds the alignment markers which our program uses to correctly orient the image to our grid
    If we changed our template to another value, we would simply add the option to our dictionary and the add
    the image to our flurorescent_templates directory

    :param image: image to match template to
    :param template: input option to determine the template to use
    :return:
    '''


    template_dictionary = {
        'template_A': 'checker_A.tif',
        'template_B': 'checker_B.tif',
    }

    if template == 'template_A':
        partition = 'A'
    else:
        partition = 'B'

    template = cv2.imread('alignment_templates/' + template_dictionary[template], cv2.IMREAD_GRAYSCALE)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if partition == 'B':
        gray_image = gray_image[0:2400,1225:2450]

    w,h = template.shape[::-1]

    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)


    deltaX = bottom_right[0] - top_left[0]
    deltaY = bottom_right[1] - top_left[1]
    midXPoint = top_left[0] + deltaX//2
    midYPoint = top_left[1] + deltaY//2


    #cv2.rectangle(image, top_left, bottom_right, 255, 2)
    #cv2.circle(image, (midXPoint, midYPoint), 80, (255, 255, 0), 2)
    #cv2.circle(image, (midXPoint, midYPoint), 2, (255, 255, 0), 2)
    if partition == 'B':
        midXPoint = midXPoint + 1225

    return (midXPoint, midYPoint)






def findAngle(alignA, alignB):
    '''

    This function finds the corresponding angle between the two top alignment markers, and also returns which
     direction they should be turned to be on the same axis.
    I could probably remake this to just return either a positive or negative angle, but this works for now
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






def rotateAndScale(img, scaleFactor = 1, degreesCCW = 0):
    '''

    :param img: the image that will get rotated and returned
    :param scaleFactor: option to enlarge, we always use at 1
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






def alignImage(image, count):
    '''
    This function combines the shiftBy function and the rotateImage function into one.
    Essentially places our image on our predetermined grid. First it rotates the image,
    and then it finds the new coordinates for Alignment Marker A. Using these new coordinates,
    it shifts the entire image such that the new Alignment Marker A coordinates are in the spot we want them to be.

    FOR THIS SETUP, WE WANT ALIGNMENT MARKER A TO BE ON COORDINATES: -> (296, 291)

    :param image: image to be aligned
    :return: shifted and rotated image
    '''


    # Rotates the image
    alignA = matchTemplate(image, "template_A")
    alignB = matchTemplate(image, "template_B")
    angle = findAngle(alignA, alignB)
    if angle > 45:
        print(angle)
        print(str(count) + ' bad image...')
    rotated_image = rotateAndScale(image, 1, angle)

    # Shifts the image
    new_alignA = matchTemplate(rotated_image, "template_A")
    alignAX = new_alignA[0]
    alignAY = new_alignA[1]

    shifted_and_rotated = shiftBy(591 - alignAX, 528 - alignAY, rotated_image)


    return shifted_and_rotated






def drawCirclesAndLabels(already_aligned_image, pointMap):
    '''
    This function is just to display the image with the labels that we predetermined,
    it has no impact on the resulting calculations
    :param image: The image that is going to be drawn on. This is the image that is ALREADY ALIGNED.
    :param pointMap:
    :return:
    '''

    copyImage = copy.deepcopy(already_aligned_image)

    for key, value in pointMap.items():
        if key not in ['A','B','C','D']:
            cv2.circle(copyImage, value, 60, (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        color = (255, 255, 255)
        thickness = 2

        copyImage = cv2.putText(copyImage, key, value, font,
                            fontScale, color, thickness, cv2.LINE_AA)
    return copyImage






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






def findAllCircleAveragesFor(imagePath, displayCirclesBool, count, folder):
    '''

    :param imagePath: the image path for the image that we are going to find all the averages for
    :param displayCirclesBool: Determines if the images will be displayed on the screen, these images
    are labeled as their corresponding number (or letter if it is an alignment marker) and the circles will be outlined
    :return:
    '''

    spots = [(591,528),(1949,528),(591,1872),(1949,1872),(1100,700),(1430,700),(760,1030),(1090,1025),(1425,1025),(1765,1032),(760,1365),(1090,1360),(1425,1360),(1760,1360),(1090,1695),(1425,1695)]

    pointMap = {
        'A': spots[0],
        'B': spots[1],
        'C': spots[2],
        'D': spots[3],
        '1': spots[4],
        '2': spots[5],
        '3': spots[6],
        '4': spots[7],
        '5': spots[8],
        '6': spots[9],
        '7': spots[10],
        '8': spots[11],
        '9': spots[12],
        '10': spots[13],
        '11': spots[14],
        '12': spots[15]
    }

    #Crops image and aligns it to our grid
    image = cv2.imread(imagePath)
    image = cropImage(image)
    aligned_image = alignImage(image, count)
    label = 'spot' + str(folder) + '_' + str(count)

    # If we choose, the aligned image with labels will
    # pop up on screen to ensure that circles are on correct points
    if displayCirclesBool == True:
        labeled_image = drawCirclesAndLabels(aligned_image, pointMap)
        cv2.imshow("Labeled Circles for " + imagePath, labeled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        labeled_image = drawCirclesAndLabels(aligned_image, pointMap)
        cv2.imwrite('/Users/quinnmaloney/Desktop/VFA_computer_vision-master/Processed/spot' + label + '.tif', labeled_image)


    #Prints the path and afterwards displays the average for each circle.
    print(imagePath)

    aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    h, w = aligned_image.shape[:2]

    output = [label] #construct output array

    for key, value in pointMap.items():

        #We do not need to print the average intensity for the alignment markers
        if key not in ['A', 'B', 'C', 'D']:

            radius_of_mask = 50
            centerPoint = value
            mask = create_circular_mask(h, w, centerPoint, radius_of_mask)
            maskedImage = np.multiply(aligned_image, mask)

            averageIntensity = findAverageLightIntensity(maskedImage, mask)
            output.append(averageIntensity)
#print(key + ": " + str(averageIntensity))
    return output





def averagesOfAllImages(displayCirclesBool = False):
    '''
    This function simply runs the findAllCircleAveragesFor every image in our list. The list is compiled by looking
    into the "fluorescent/" directory and selecting the files that begin with "image" as the file name. This is
    to prevent any other types of files to get passed in. This also means that any test image must be named
    as "image*". It just has to start with the word image.

    :param displayCirclesBool:
    :return:
    '''
    name = 'tiff-conv '
    folders = 1
    for folder in range(1,folders+1):
        mypath = name + str(folder) + '/'
        imageList = [f for f in listdir(mypath) if (isfile(join(mypath, f))) and ''.join(f[3]) == '-']
        imageList = sorted(imageList)
        print(str(len(imageList)) + ' images imported...')
        #print(imageList)
        #imageList = ['image_1.tif', 'image_2.tif', 'image_3.tif', 'image_4.tif', 'image_5.tif', 'image_6.tif']
        i = 0
        matrix = np.ones(13)
        for image in imageList:
            if i==0:
                matrix = findAllCircleAveragesFor(mypath + image, displayCirclesBool,i,folder)
                i += 1
                continue
            matrix = np.vstack([matrix,findAllCircleAveragesFor(mypath + image, displayCirclesBool, i,folder)])
            i += 1
        with open(name + str(folder) + '.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerows(matrix)



def main():
    #Change to true to display images with circles drawn on
    averagesOfAllImages(False)

if __name__ == '__main__':
    main()




'''
This is a list of where all the circles are located in our fixed grid. This is how I originally found them
This is now irrelevant, but here for reference

cv2.circle(aligned_image, (690, 440), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (985, 440), 60, (255, 155, 70), 2)

cv2.circle(aligned_image, (445, 665), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (690, 665), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (985, 665), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (1225, 665), 60, (255, 155, 70), 2)

cv2.circle(aligned_image, (835, 800), 60, (255, 155, 70), 2)

cv2.circle(aligned_image, (445, 935), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (690, 935), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (985, 935), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (1225, 935), 60, (255, 155, 70), 2)

cv2.circle(aligned_image, (690, 1170), 60, (255, 155, 70), 2)
cv2.circle(aligned_image, (985, 1170), 60, (255, 155, 70), 2)
'''


