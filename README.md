PandaVision

This repo provides a script that can be used to find Aruco markers on a table and return them 
in the robot's coordinate frame. 

Required packages:

    numpy
    cv2-contrib (https://pypi.org/project/opencv-contrib-python/)

Usage (or see the main function as example in main.py):

1. Instantiate ObjectFinder with `object_finder = ObjectFinder()`
2. call `object_finder.find_objects(img, world_z=0)` where img is a BGR image in opencv format and 
world_z is the height of the markers in mm from the robot's z plane. For objects that are less tall 
than ~20mm this can be left at 0 as the error is minimal. This function will return a dictionary of objects, 
where each object contains all the relevant data for the marker. See line 21-41 for data fields.