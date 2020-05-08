Uses fluorescent_tiff_final.py to circle immunoreaction spots and find pixel averages.

To scan images in N folders titled 'name_of_folders n' for 0<n<=N:
    -SET VARIABLES 'name' and 'folders' in averagesOfAllImages on lines 409 and 410, respectively.
    -THEN SET imageList parameters i and c in line 413, shown here -->  and ''.join(f[i]) == 'c' <-- where i is the index of the character c in each image name. For example, if all images being scanned are labeled xxx-xxxx-xxx i and c could be 3 and '-', or 8 and '-'.
    -Circled images are stored to 'Processed/'.
    -The .csv's are stored in the main directory.
    
Change template
    -Template aligner images are stored in a directory named 'alignment_templates/'.
    -Spot locations are stored in the 'spots' array of findAllCircleAveragesFor (the first four are alignment markers).
    -Change circle radius in the 'radius_of_mask' integer in findAllCircleAveragesFor.
    -The current template is based on 04/27 images.
