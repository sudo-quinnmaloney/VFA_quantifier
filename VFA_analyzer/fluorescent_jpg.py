'''
HOW THIS PROGRAM WORKS ON A HIGH LEVEL:
1. First it crops the image to our specified dimensions, to which right now it is set to 2540x2400 -> width x height
2. Then it uses template matching in order to find the top two alignment markers, and using the points
of the alignment markers, it rotates the image and scales the image if found necessary.
3. Then it shifts the image to where the first alignment marker, Alignment Marker A, is at the spot we
need it to be now that the image is upright and oriented correctly. In our case, we want that alignment
marker A to be at the point (591, 528). Once the image is aligned, we isolate each immunoreaction spot. By calculating
the centroid we can greatly increase the accuracy of our mask alignment.
4. Once the image is aligned, we make a mask for each individual circle, and multiply (element-wise) it by the original
image to create a new image that outside of the mask, it is completely black (matrix value of 0) We calculate
the average light intensity by taking the sum of the image value (inside the mask, the values will remain
their original values) divided by the sum of the mask
(this is essentially just the area of the circle because inside the mask,
there are only 1's and outside it there are only 0's)
5. We repeat this process for every image in the directory the user specifies, which houses all
 our images that need analysis. As long as our directory specified is inside the datasets directory,
 it will work as expected


 VERY IMPORTANT NOTE: to change the map we have, and make
 the program still work, the only things that should be changed should be one of the following:

    1. pointMap
    2. template_dictionary
    3. MASK_RADIUS
    4. YMIN_BOUND
    5. YMAX_BOUND
    6. XMIN_BOUND
    7. XMAX_BOUND

By changing these values, we should be able to make our code work with any changes that we want to make.
i.e. If you have a new image that you have to change the cropping to (maybe bc image is way larger or smaller)
    or you want to change the map, changing the values described above should be the only changes
    you make in order to make program still work

However, if you change the image type (e.g. from tiff to jpg) you will need to
change the code in the functions below

There might be other corner cases where you have to change the code in the functions, but generally,
you should be able to change just these constants

'''

import cv2
import numpy as np
import csv
import time
from helper_functions_v2 import drawCirclesAndLabels, \
    alignImage, \
    localizeWithCentroid, getStats

#This is for reading the images that are in the fluorescent/ directory
from os import listdir, mkdir
from os.path import isfile, join, isdir

ACOORD = (1012, 1112)
BCOORD = (2340, 1112)
CCOORD = (1012, 2442)
DCOORD = (2340, 2442)




#### THIS PART IS ESSENTIAL, THIS IS OUR SPOT MAP THAT OUR ENTIRE PROGRAM IS BASED ON
pointMap = {
    'A': (1012, 1112),
    'B': (2340, 1112),
    'C': (1012, 2442),
    'D': (2340, 2442),
    '1': (1515, 1280),
    '2': (1840, 1280),
    '3': (1185, 1610),
    '4': (1515, 1610),
    '5': (1840, 1610),
    '6': (2170, 1610),
    '7': (1185, 1935),
    '8': (1515, 1935),
    '9': (1840, 1935),
    '10': (2170, 1935),
    '11': (1515, 2265),
    '12': (1840, 2265),
    '13' : (1676, 1777)
}

#Note: if you change this, only change the file name of
# the template, dont change the keys of this dictionary
# e.g. Do not change 'template_A'
template_dictionary = {
    'template_A': 'alignment_A_CPN.jpg',
    'template_B': 'alignment_B_CPN.jpg',
    'template_C': 'alignment_C_CPN.jpg',
    'template_D': 'alignment_D_CPN.jpg'
}


# CONSTANTS
MASK_RADIUS = 60
ALIGNMENT_MARKER_A_MAP_LOCATION = pointMap['A']
ALIGNMENT_MARKER_B_MAP_LOCATION = pointMap['B']
DISTANCE_FROM_A_TO_B = ALIGNMENT_MARKER_B_MAP_LOCATION[0] - ALIGNMENT_MARKER_A_MAP_LOCATION[0]

#CROP IMAGE DIMENSIONS CONSTANTS
YMIN_BOUND = 600
YMAX_BOUND = 4200
XMIN_BOUND = 200
XMAX_BOUND = 3800





def getCircleData(imagePath, image_name, displayCirclesBool, whichCommand):
    '''
    :param imagePath: the image path for the image that we are going to find all the averages for
    :param image_name: This is the name of the image that we are going to find all the averages for
    :param displayCirclesBool: Determines if the images will be displayed on the screen, or saved to a file

        True: The images will be output to the screen

        False: The images will be saved to a directory named "processed" inside
        the directory the user specified at the beginning


    :return:
    '''




    ##### This output array will be returned and will be a row in the csv file
    output = []
    output.append(image_name)





    #### Crops image and aligns it to our grid
    full_image_path = imagePath + image_name
    image = cv2.imread(full_image_path)
    image = image[YMIN_BOUND: YMAX_BOUND, XMIN_BOUND: XMAX_BOUND]
    aligned_image = alignImage(image, image_name, DISTANCE_FROM_A_TO_B, ALIGNMENT_MARKER_A_MAP_LOCATION, template_dictionary)

    ##So that we can create a new image name with _processed appended to it
    image_name = image_name.split('.')[0]

    for i in range(13):
        try:
            pointMap[str(i+1)] = localizeWithCentroid(aligned_image, pointMap[str(i+1)],str(i+1), False)
        except:
            continue



    #### Either displays the result images on the screen or saves them to directory inside calling directory
    if displayCirclesBool == True:
        labeled_image = drawCirclesAndLabels(aligned_image, pointMap, MASK_RADIUS)
        cv2.imshow("Labeled Circles for " + image_name, labeled_image)

    else:
        labeled_image = drawCirclesAndLabels(aligned_image, pointMap, MASK_RADIUS)
        if not isdir(imagePath + 'processed_jpg/'):
            mkdir(imagePath + 'processed_jpg/')
        cv2.imwrite(imagePath + 'processed_jpg/' + image_name + '_processed.jpg', labeled_image)







    ## Grayscale the image to begin masking process
  #  aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    aligned_image = aligned_image[:,:,2]
    h, w = aligned_image.shape[:2]

    for key, value in pointMap.items():

        #We do not need to print the average intensity for the alignment markers
        if key not in ['A', 'B', 'C', 'D']:

            #mask = create_circular_mask(h, w, MASK_RADIUS, value[0], value[1])
            #maskedImage = np.multiply(aligned_image, mask)

            #averageIntensity = findAverageLightIntensity(maskedImage, mask)
            output.append(getStats(aligned_image, MASK_RADIUS, value, whichCommand))



    #Prints the path for the image that was just processed
    print("Finished Processing: " + full_image_path)

    return output



def averagesOfAllImages(displayCirclesBool = False):
    '''

    This function simply runs the findAllCircleAveragesFor every image in our list.

    The list is compiled by first prompting the user for the name of the directory they would like to run a test on.
    It then finds all the tif images inside of that directory and adds them to the list.

    NOTE: THE REPOSITORY THE USER ENTERS MUST BE INSIDE 'datasets' DIRECTORY.

    For each image, once it receives the intensity of each spot, it takes the information
    and writes it to a csv file inside of the user-specified directory.

    E.g. Say that we have a directory named 'tiff-conv1' inside the 'datasets' directory
        Once the processing is done, inside 'datasets/tiff-conv1' there will be a csv file named 'tiff-conv1.csv'
        containing all the informatino that we found


    :param displayCirclesBool:
    :return:
    '''
    imageList = []
    
    while(1):
        #### User specified test directory
        test_directory_name = input('Enter directory to test, or \'quit\' to exit: ')
        if test_directory_name == 'quit':
            return
        if test_directory_name[-1] != '/':
            test_directory_name += '/'
        test_directory_path= test_directory_name


        ##Asserting that the directory input by user is valid and has images ending with .tif inside of it
        if(isdir(test_directory_path)):
            imageList = [f for f in listdir(test_directory_path) if (isfile(join(test_directory_path, f))) and (f.endswith('.jpg') or f.endswith('.jpeg'))]
            if (len(imageList) > 0):
                break
            else:
                print("\tError: No jpg/jpeg images in this directory, please check the directory")
        else:
            print("\tError: Invalid directory")


    imageList = sorted(imageList)
    print(str(len(imageList)) + ' images imported...')
    
    commands = ['std','mean','max','min']
    stat_command = ""
    while (1):
        stat_command = input('Enter desired metric (\'std\', \'mean\', \'max\',\'min\'): ').lower()
        if stat_command in commands:
            break;
        else:
            print('\tInvalid metric...')
    command = commands.index(stat_command)
    
    #testImage[:, :, 1]
    start = time.time()
    ##### Writes data acquired from list to our csv file
    i = 0
    matrix = np.ones(13)
    for image in imageList:
        if i == 0:
            matrix = [getCircleData(test_directory_path, image, displayCirclesBool, command)]
            i += 1
            continue
        matrix = np.vstack([matrix,getCircleData(test_directory_path, image, displayCirclesBool, command)])
        i += 1
    if not isdir(test_directory_path + 'csv_jpg/'):
        mkdir(test_directory_path + 'csv_jpg/')
    with open(test_directory_path + 'csv_jpg/' + stat_command + '.csv', 'w+', newline='') as f:
        #matrix = np.vstack([matrix, np.ones(13)])
        writer = csv.writer(f, delimiter = ';')
        writer.writerows(matrix)
    end = time.time()
    print('Average runtime: ' + str((end - start)/len(imageList)))


def main():
    
    #Change to true to display images with circles drawn on
    averagesOfAllImages(False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
