# TrainAndTest.py

import cv2
import numpy as np
import operator
import os
import imutils
import time
import math
def nothing(x):
    pass
cv2.namedWindow('Resultado')
#h,s,v = 0,0,0	
# Creating track bar
#cv2.createTrackbar('h', 'result',0,179,nothing)
#cv2.createTrackbar('s', 'result',0,255,nothing)
#cv2.createTrackbar('v', 'result',0,255,nothing)
#cv2.createTrackbar('c1', 'result',0,179,nothing)
#cv2.createTrackbar('c2', 'result',0,255,nothing)
#cv2.createTrackbar('c3', 'result',0,255,nothing)
# module level variables ##########################################################################
MIN_CONTOUR_AREA = 1000
MAX_CONTOUR_AREA = 20000

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA or self.fltArea > MAX_CONTOUR_AREA: 
			#print(self.fltArea)
			return False        # much better validity checking would be necessary
        #print(self.fltArea)
        return True

###################################################################################################
while(1):
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("classifications_plate.txt", np.float32)                  # read in training classifications
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images_plate.txt", np.float32)                 # read in training images
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        
    # end try
    
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    
    
    img = cv2.imread("nmr.JPG")          # read in testing numbers image
    img3 = img
    img3 = cv2.resize(img3, (500, 200))
    img4 = cv2.resize(img3, (500, 200))
    min_area = 5000
    max_area = 50000
    cont_filtered = []    
    
    #h = cv2.getTrackbarPos('h','result')
    #s = cv2.getTrackbarPos('s','result')
    #v = cv2.getTrackbarPos('v','result')
    
    lower_limit = np.array([0,0,70])
    upper_limit = np.array([180,255,255])

    
    kernel = np.ones((2,2),np.uint8)
    kernel_lg = np.ones((2,2),np.uint8)
    imgTestingNumbers = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    imgGray =cv2.inRange(imgTestingNumbers, lower_limit, upper_limit)
    
    mask = imgGray
    mask2 = img4
    imgTestingNumbers = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)
    img = cv2.bitwise_and(img3,img3,mask = mask)
    thresh = mask
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # filter out all contours below a min_area
    for cont in contours:
        if cv2.contourArea(cont) > min_area and cv2.contourArea(cont) < max_area:
            cont_filtered.append(cont)
            #print(cv2.contourArea(cont))

    # just take the first contour (we're assuming we only have one here)
    # the try\except is necessary, since the program will crash if it tries to access 
    # contours[0] and there aren't any
    try:
        cnt = cont_filtered[0]
        hull = cv2.convexHull(cnt)
        # draw the rectangle surrounding the filtered contour
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[hull],0,(0,0,255),2)
        
        
        rows,cols = thresh.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)        
        
        # this would draw all the contours on the image, not just the ones from cont_filtered
        cv2.drawContours(img, cont_filtered, -1, (0,255,0), 3)
        rows,cols,ch = img.shape
        #pts1 = np.float32([[665,300],[1852,360],[650,640],[1852,723]])
        pts1 = np.float32([[box[2][0],box[2][1]],[box[3][0],box[3][1]],[box[1][0],box[1][1]],[box[0][0],box[0][1]]])
        pts2 = np.float32([[-100,-50],[800,-50],[-100,800],[800,750]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        img4 = cv2.warpPerspective(img4,M,(700,400))
        img4 = cv2.resize(img4, (1200, 300))

    except:
        print('no contours')   
    
    
    #h2 = cv2.getTrackbarPos('c1','result')
    #s2 = cv2.getTrackbarPos('c2','result')
    #v2 = cv2.getTrackbarPos('c3','result')
    lower_limit2 = np.array([0,0,60])
    upper_limit2 = np.array([180,255,255])
    mask2 = cv2.inRange(img4, lower_limit2, upper_limit2)
    cv2.bitwise_not ( mask2, mask2 )
    imgBlurred = cv2.GaussianBlur(mask2, (9,9), 0)      # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program
    i = 0
    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(img4,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results
        
        strFinalString = strFinalString + strCurrentChar            # append current char to full string
        i =i +1
    # end for
	str1 = strFinalString
	list1 = list(str1)
    if str1[0] == '4' : 		
		list1[0] = 'A'
		str1 = ''.join(list1)
    print "\n" + "Placa numero: "+ str1 + "\n"                  # show the full string

    mask2 = cv2.resize(mask2, (600, 200))
    img4 = cv2.resize(img4, (600, 200))
    cv2.imshow("Tratativa", mask2)
    cv2.imshow("Perspectiva", img4)
    cv2.imshow("Resultado", img3)      			# show input image with green boxes drawn around found digits  
    cv2.waitKey(5)                              # wait for user key press



###################################################################################################
if __name__ == "__main__":
    main()
# end if









