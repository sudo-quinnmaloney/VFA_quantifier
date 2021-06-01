Uses quantify_VFA.py to circle immunoreaction spots and find pixel averages, standard means, minima, and maxima

To use:
    -The script will wait for user input. Enter a the name of a folder within the script's parent directory.
    -In addition to the folder name, parameters can be given when separated by a comma. These include:
        -Toggle desired metrics in the form of a binary number. Type enter 'help' in the script for more details.
        -Set desired radius of analysis. Type 'help' for instructions.
        -Toggle display boolean for individual spot masks.
    -Alternatively, call the script using " python3 quantify_VFA.py <<< $'example_folder,1111,r=49\n' " to skip manual input
    -Notated images and .csv's are stored in sibling folders.
    
Change template
    -Template aligner images are stored in a directory named 'alignment_templates/'.
    -Spot locations are stored in the global 'pointMap' dictionary of quantify_VFA.py (the first four are alignment markers).
    -NUM_SPOTS should be changed to match current VFA designs
