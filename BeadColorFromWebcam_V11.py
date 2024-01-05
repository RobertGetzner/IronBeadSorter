"""
Desc: This program is used to detect the color of iron beads from a webcam.
The iron beads are displayed in different angles, distances, colors, light conditions, sizes, shapes, and textures.
Challenges:
- Limited knowledge of Python, OpenCV, and image processing (Co Pilot does the coding).
- Variations in lighting and captured images make comparison difficult.
Basic logic of the program:
1. Get the image from the webcam connected via USB.
2. Show different iron beads to the camera.
3. Detect the color of each iron bead.
4. Continuously display the detected color.
The program has different parts:
1. Find all cameras connected to the computer to identify the correct camera.
2. Create training pictures or classify images/beads during the actual 'sorting'.
3. Capture training images from webcam 1 and store them to disk.
4. Find the iron beads in the captured images and send "m" to the serial port when a bead is detected.
5. Determine the color class by comparing the captured image with the training images.
Further general remarks:
- Some analysis/logging parts are disabled for better performance.
- Some parts of the code are taken from the internet.
"""

# Import the necessary packages
import numpy as np
import imutils
import cv2
import time
import sys
import os
import matplotlib
import serial
from matplotlib import pyplot as plt
import collections
from collections import Counter
import winsound

# set to true to create training images without rotating the wheel with bins
# training = True 
training = False

# set to true to find the camera number
# findCamera = True
findCamera = False

# camera number
cameraNumber = 0

# start time for serial communication "waiting for command" timeout
wfc_timeout_lasttime = time.time() 
wait4cmd_timeout = 0.5 # 1  # waiting for command timeout in seconds
tresholdProbeOf3 = 0.6

# Find cameras connected to the computer to determine the correct camera number
if (findCamera):
    i = 0
    while i < 20:
        try:
            cv2.namedWindow("preview") 
            vc = cv2.VideoCapture(i)

            if vc.isOpened():
                rval, frame = vc.read()
            else:
                rval = False
            while rval:
                cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.imshow("preview", frame)
                rval, frame = vc.read()
                key = cv2.waitKey(20)
                if key == 27:
                    break
            cv2.destroyWindow("preview")
            
        except Exception as exCamError:
            print("error: " + str(exCamError))
        i += 1


# start serial port
def start_serial_port():
    try:
        ser = serial.Serial('COM5', 115200, timeout=1, write_timeout=5)
    except Exception as ex:
        print("\033[91m" + "error: " + str(ex) + "\033[0m")
        sys.exit()
    return ser

# Load all training images from disk and store them in suitable global data structure
trainingImages = dict()
def loadTrainingImages():
    """
    Load all training images from disk and store them in a data structure.
    The training images are stored in a folder with a specific filename prefix.
    Each color class has a separate folder with images.
    """
    filenameCounter = 0
    for filename in os.listdir("C:\\Users\\robert\\Documents\\pythonRobert\\IronBeadsColorDetect\\trainingsImages"):
        image = cv2.imread("C:\\Users\\robert\\Documents\\pythonRobert\\IronBeadsColorDetect\\trainingsImages\\" + filename)
        fn = filename[8:]
        fn = fn[:-4]
        fn = fn.replace("(", "")
        fn = fn.replace(")", "")
        colorClass = fn.rstrip('0123456789')
        colorClass = colorClass.lower()
        if colorClass not in trainingImages:
            trainingImages[colorClass] = []
        trainingImages[colorClass].append(image)
        filenameCounter += 1

# define a function to find circles in a training image
# the function has a parameter with the frame to be analyzed and returns a mask to leaving the 
# parts of the image outside the circle black
def findCircleInTrainingImage(iv_frame, iv_mask_r_out=110, iv_mask_r_in=70, iv_maxRadius=120, iv_minRadius=100):
    x = y = 0
    # create white mask to show full image
    mask = np.zeros((iv_frame.shape[0], iv_frame.shape[1]), dtype="uint8")
    mask = cv2.bitwise_not(mask)

    # convert the image to grayscale
    gray = cv2.cvtColor(iv_frame, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 1400, iv_maxRadius, iv_minRadius)
    # ensure at least some circles were found, if so take another image
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        try:
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                # cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(iv_frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        except Exception as ex:
            print("error: " + str(ex))

        # create a black circle mask with center x,y and radius 200
        if (x > 75 and y>0 and r > 60):
            mask = np.zeros((iv_frame.shape[0], iv_frame.shape[1]), dtype="uint8")
            cv2.circle(mask, (x, y), iv_mask_r_out, 255, -1)
            # add the mask to the image
            # frame = cv2.bitwise_and(frame, frame, mask=mask)
            # show the mask
            # cv2.imshow("mask", mask)
            # create the same mask again but with radiu 40 and add it to the image
            mask2 = np.zeros((iv_frame.shape[0], iv_frame.shape[1]), dtype="uint8")
            cv2.circle(mask2, (x, y), iv_mask_r_in, 255, -1)
            # invert mask2
            mask2 = cv2.bitwise_not(mask2)
            # combine mask and mask2
            mask = cv2.bitwise_and(mask, mask2)
    return mask

# Define a function to create a data structure to store the histogram of each training image per color class    
# for each trainnig image search for circles in the image and create a mask 
# use the mask in the histogram calculation
def createTrainingImageHistograms():
    # load the training images from disk using the function loadTrainingImages()
    loadTrainingImages()
    # create a data structure to store the histograms of the training images
    trainingImageHistograms = dict()
    # loop over all color classes
    show_training_images = False
    for colorClass in trainingImages:
        # create a list to store the histograms of the training images for the color class
        trainingImageHistograms[colorClass] = []
        # loop over all training images for the color class
        for trainingImage in trainingImages[colorClass]:   
            # convert the training image to HSV color space
            trainingImageHSV = cv2.cvtColor(trainingImage, cv2.COLOR_BGR2HSV)    
            # find circles in the image
            mask = findCircleInTrainingImage (trainingImageHSV, iv_mask_r_out=110, iv_mask_r_in=70, iv_maxRadius=120, iv_minRadius=100)
            # apply mask to frame_circle
            trainingImageHSV = cv2.bitwise_and(trainingImageHSV, trainingImageHSV, mask=mask)

            # calculate the histogram of the training image using the mask
            trainingImageHist = cv2.calcHist([trainingImageHSV], [0, 1], mask, [180, 256], [0, 180, 0, 256])
            # store the histogram of the training image in the data structure
            trainingImageHistograms[colorClass].append(trainingImageHist)

            if (show_training_images):
                # show the RGBframe
                trainingImage = cv2.bitwise_and(trainingImage, trainingImage, mask=mask)
                # add information text to the image: color class
                cv2.putText(trainingImage, "Training: " + colorClass, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("FrameTrainingImg", trainingImage)
                # get a key, if key is ESC, stop the loop
                key = cv2.waitKey(1000) & 0xFF
                if key == 27:
                    show_training_images = False
    return trainingImageHistograms


# define a function to find the color class of a captured image using training image histograms
def findColorClassUsingHistograms(capturedImage, iv_mask, trainingImageHistograms):
    """
    Find the color class of the iron bead in the captured image by comparing it with training images.
    The captured image is compared with all images in the training set.
    The similarity between the captured image and each training image is calculated using histograms.
    The most probable color class is determined based on the highest similarity.
    """
    mostProbableColorClass = ""
    highestSimilarity = 0
    capturedImageHSV = cv2.cvtColor(capturedImage, cv2.COLOR_BGR2HSV)
    capturedImageHist = cv2.calcHist([capturedImageHSV], [0, 1], iv_mask, [180, 256], [0, 180, 0, 256])
    # capturedImageHist = cv2.calcHist([capturedImageHSV[np.where(val > 0,True, False)]], [0, 1], None, [180, 256], [0, 180, 0, 256])
    for colorClass in trainingImageHistograms:
        for trainingImageHist in trainingImageHistograms[colorClass]:
            similarity = cv2.compareHist(capturedImageHist, trainingImageHist, cv2.HISTCMP_CORREL)
            if similarity > highestSimilarity:
                highestSimilarity = similarity
                mostProbableColorClass = colorClass
    return mostProbableColorClass, highestSimilarity

# Define a function to find the color class of a captured image
def findColorClass(capturedImage ,iv_mask):
    """
    Find the color class of the iron bead in the captured image by comparing it with training images.
    The captured image is compared with all images in the training set.
    The similarity between the captured image and each training image is calculated using histograms.
    The most probable color class is determined based on the highest similarity.
    """
    mostProbableColorClass = ""
    highestSimilarity = 0
    capturedImageHSV = cv2.cvtColor(capturedImage, cv2.COLOR_BGR2HSV)
    capturedImageHist = cv2.calcHist([capturedImageHSV], [0, 1], iv_mask, [180, 256], [0, 180, 0, 256])
    # capturedImageHist = cv2.calcHist([capturedImageHSV[np.where(val > 0,True, False)]], [0, 1], None, [180, 256], [0, 180, 0, 256])
    for colorClass in trainingImages:
        for trainingImage in trainingImages[colorClass]:
            # make a copy of the training image
            trainingImage2 = trainingImage.copy()
            # in trainingsimage2, set all pixel to white, which are not black
            trainingImage2[np.where((trainingImage2 != [0,0,0]).all(axis = 2))] = [255,255,255]
            mask3 = trainingImage2[:,:,0]

            trainingImageHSV = cv2.cvtColor(trainingImage, cv2.COLOR_BGR2HSV)
            TrainingImageHist = cv2.calcHist([trainingImageHSV], [0, 1], mask3, [180, 256], [0, 180, 0, 256])
            # TrainingImageHist = cv2.calcHist([trainingImageHSV[np.where(val > 0,True, False)]], [0, 1], None, [180, 256], [0, 180, 0, 256])
            similarity = cv2.compareHist(capturedImageHist, TrainingImageHist, cv2.HISTCMP_CORREL)
            # if colorClass == "empty": # and similarity < 0.2:
            #     similarity = 0   # empty must be very similar to count as empty
            if similarity > highestSimilarity:
                highestSimilarity = similarity
                mostProbableColorClass = colorClass
    return mostProbableColorClass, highestSimilarity

# define a function to wait for commmand text from arduino on serial port
# return line read from serial port
def wait_for_arduino_ready4cmd_withtimeout(ser):
    line = "" 
    readline = ""
    wfc_timeout_lasttime = time.time() 
    while ( line != b"waiting for command\r\n" and (time.time()-wfc_timeout_lasttime) < wait4cmd_timeout): # line != b"bin reached\r\n" and            
        try:
            # read from serial port
            readline = ser.readline()
            # print line to console
            if (readline != b"waiting for command\r\n"):
                print(readline)
            if (readline!=b''):
                line = readline
        except Exception as ex:
            # print error to console in red color
            print("\033[91m" + "error: " + str(ex) + "\033[0m")
    return line

# define a function to send command to arduino on serial port with bin number
# the function has one parameter: color class
def send_command_with_bin_number(ser, iv_colorClass):
    if (1==0):  # farb block 0                        
        if "pink_" == iv_colorClass:
            ser.write(b'b1\n')
        elif "pink_dark" == iv_colorClass:
            ser.write(b'b2\n')  
        elif "yellow" == iv_colorClass:
            ser.write(b'b3\n')
        elif "yellow_light" == iv_colorClass:
            ser.write(b'b4\n')
        elif "blue_light" == iv_colorClass:
            ser.write(b'b5\n')  
        elif "orange" == iv_colorClass: 
            ser.write(b'b6\n')
        elif "green" == iv_colorClass:
            ser.write(b'b7\n')
        elif "green_medium" == iv_colorClass:  
            ser.write(b'b8\n')
        elif "red" == iv_colorClass:
            ser.write(b'b9\n')
        elif "skin" == iv_colorClass:
            ser.write(b'b10\n')
        elif "brown" == iv_colorClass:
            ser.write(b'b11\n')
        elif "violet" == iv_colorClass:
            ser.write(b'b12\n')
        elif "violet_dark" == iv_colorClass:
            ser.write(b'b13\n')
        else:
            ser.write(b'b0\n')
    if (1==0): # color block 1     
        if "skin" == iv_colorClass:
            ser.write(b'b1\n')
        elif "grey_light" == iv_colorClass:
            ser.write(b'b2\n')  
        elif "white" == iv_colorClass:
            ser.write(b'b3\n')
        elif "pink_dark" == iv_colorClass:
            ser.write(b'b4\n')
        elif "clear" == iv_colorClass:
            ser.write(b'b5\n')  
        elif "pink" == iv_colorClass: 
            ser.write(b'b6\n')
        elif "violet_dark" == iv_colorClass:
            ser.write(b'b7\n')
        elif "violet" == iv_colorClass:  
            ser.write(b'b8\n')
        elif "red" == iv_colorClass:
            ser.write(b'b9\n')
        elif "green_light" == iv_colorClass:
            ser.write(b'b10\n')
        elif "brown" == iv_colorClass:
            ser.write(b'b11\n')
        elif "grey" == iv_colorClass:
            ser.write(b'b12\n')
        elif "black" == iv_colorClass:
            ser.write(b'b13\n')
        else:
            ser.write(b'b0\n') 
            # play a beep sound
            # winsound.Beep(1000, 200)
              
    if (1==0): # color block 2                                       
        if "grey_light" == iv_colorClass:
            ser.write(b'b1\n')
        elif "yellow_light" == iv_colorClass:
            ser.write(b'b2\n')  
        elif "orange" == iv_colorClass:
            ser.write(b'b3\n')
        elif "yellow" == iv_colorClass:
            ser.write(b'b4\n')
        elif "blue_light" == iv_colorClass:
            ser.write(b'b5\n')  
        elif "blue" == iv_colorClass: 
            ser.write(b'b6\n')
        elif "green_dark" == iv_colorClass:
            ser.write(b'b7\n')
        elif "green_medium" == iv_colorClass:  
            ser.write(b'b8\n')
        elif "red" == iv_colorClass:
            ser.write(b'b9\n')
        elif "green_light" == iv_colorClass:
            ser.write(b'b10\n')
        elif "white" == iv_colorClass:
            ser.write(b'b11\n')
        elif "grey" == iv_colorClass:
            ser.write(b'b12\n')
        elif "black" == iv_colorClass:
            ser.write(b'b13\n')
        else:
            ser.write(b'b0\n')                               
    if (1==0): # color block 3                                       
        if "pink_" == iv_colorClass:
            ser.write(b'b1\n')
        elif "yellow_light" == iv_colorClass:
            ser.write(b'b2\n')  
        elif "orange" == iv_colorClass:
            ser.write(b'b3\n')
        elif "yellow" == iv_colorClass:
            ser.write(b'b4\n')
        elif "blue_light" == iv_colorClass:
            ser.write(b'b5\n')  
        elif "blue" == iv_colorClass: 
            ser.write(b'b6\n')
        elif "green_dark" == iv_colorClass:
            ser.write(b'b7\n')
        elif "green_medium" == iv_colorClass:  
            ser.write(b'b8\n')
        elif "red" == iv_colorClass:
            ser.write(b'b9\n')
        elif "green_light" == iv_colorClass:
            ser.write(b'b10\n')
        elif "white" == iv_colorClass:
            ser.write(b'b11\n')
        elif "grey_light" == iv_colorClass:
            ser.write(b'b12\n')
        elif "black" == iv_colorClass:
            ser.write(b'b13\n')
        else:
            ser.write(b'b0\n')       
    if (1==1): # color block Michaela 1
        if "mviolet" == iv_colorClass:
            ser.write(b'b1\n')
        elif "yellow_light" == iv_colorClass:
            ser.write(b'b2\n')  
        elif "morange" == iv_colorClass or "brown" == iv_colorClass:
            ser.write(b'b3\n')
        elif "yellow" == iv_colorClass:
            ser.write(b'b4\n')
        elif "violet" == iv_colorClass:
            ser.write(b'b5\n')  
        elif "myellow" == iv_colorClass: 
            ser.write(b'b6\n')
        elif "green_dark" == iv_colorClass:
            ser.write(b'b7\n')
        elif "mgreen" == iv_colorClass:  
            ser.write(b'b8\n')
        elif "red" == iv_colorClass:
            ser.write(b'b9\n')
        elif "green_light" == iv_colorClass:
            ser.write(b'b10\n')
        elif "white" == iv_colorClass:
            ser.write(b'b11\n')
        elif "gray_light" == iv_colorClass:
            ser.write(b'b12\n')
        elif "black" == iv_colorClass:
            ser.write(b'b13\n')
        else:
            ser.write(b'b0\n')    
            # play a beep sound
            # winsound.Beep(1000, 1000)
# define a function to find circles in the image
# the function has a parameter with the frame to be analyzed and returns x and y coordinates 
# of the center of the circle
# last working: iv_mask_r_out=120, iv_mask_r_in=70, iv_minradius=100, iv_maxradius=140
def findCircleDuringSorting(iv_frame, iv_mask_r_out=110, iv_mask_r_in=70, iv_minradius=100, iv_maxradius=120):
    x = y = 0
    # create white mask to show full image
    mask = np.zeros((iv_frame.shape[0], iv_frame.shape[1]), dtype="uint8")
    mask = cv2.bitwise_not(mask)

    # convert the image to grayscale
    gray = cv2.cvtColor(iv_frame, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 1400)
    # ensure at least some circles were found, if so take another image
    if circles is not None:
        # get another frame (stopped motor?)
        (grabbed, iv_frame) = camera.read()
        # resize the frame
        frame = imutils.resize(iv_frame, width=800)

        # convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 1400, minRadius=60, maxRadius=200)
        if circles is not None:   
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                # cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                
            # create a black circle mask with center x,y and radius 200
            if (x > 75 and y>0 and r > 60):
                # show the image in "readings with info"  if relevant circles were found
                cv2.imshow("Frame", iv_frame)            # find circles in the image
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")
                cv2.circle(mask, (x, y), iv_mask_r_out, 255, -1)
                # add the mask to the image
                # frame = cv2.bitwise_and(frame, frame, mask=mask)
                # show the mask
                # cv2.imshow("mask", mask)
                # create the same mask again but with radiu 40 and add it to the image
                mask2 = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")
                cv2.circle(mask2, (x, y), iv_mask_r_in, 255, -1)
                # invert mask2
                mask2 = cv2.bitwise_not(mask2)
                # combine mask and mask2
                mask = cv2.bitwise_and(mask, mask2)
                # add the mask to the image
                frame = cv2.bitwise_and(frame, frame, mask=mask)
                # show the mask
                # cv2.imshow("mask", mask)
                
    return iv_frame, x, y, mask

# main procesing 
if (1==1):
    camera = cv2.VideoCapture(cameraNumber)
    camera.set(3, 800)
    camera.set(10, 0.1)
    camera.set(11, 0.1)

    if (training):
        print("enter the filename prefix color:")
        filenamePrefix = 'Training' + input()
        filenameCounter = 0
        no_unclassified = 0 
    
    ser = start_serial_port()
    # send start command to arduino
    ser.write(b'Start\n')

    lastFoundColorClass = ""  # last which was not empty
    lastMeasuredColorClass = ""  # last which was measure including empty 
    colorClass = ""  # current color class
    lastSimilarity = 0
    highestSimilarity = 0
    x = y = 0

    colorClassStatistics = dict()
    lastMeasureTimestamp = time.time()

    # wait two seconds for camera to be ready
    time.sleep(2)

    # loadTrainingImages()
    trainingImageHistograms = createTrainingImageHistograms()	
    
    bin2Go = "b0\n"
    while True:
        # wait a key to be pressed and store the pressed key in variable key
        # stop loop if key is ESC
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        elif key == ord('0'):
            # wait for arduino to be ready
            wait_for_arduino_ready4cmd_withtimeout(ser)
            # move to bin 0 
            ser.write(b'b0\n')   
            
        wait_for_arduino_ready4cmd_withtimeout(ser)

        (grabbed, frame) = camera.read()
        # resize the frame
        frame = imutils.resize(frame, width=800)
        # show the frame
        cv2.imshow("Live-Frame", frame)  

        #--------------------------------------------------------------------------------------------
        # find circles in the image to start measuring
        if (1==1): # and (time.time() - lastMeasureTimestamp) > 0.8):
       
            # find circles in the image
            frame, x, y, mask = findCircleDuringSorting(frame)	

            # show a 2d histo plot of the grabbed frame
            if (1==0):   
                # show 2d histogram plot of frame in additional window
                # create a histogram from frame and shot a 2d histogram plot
                # hist = cv2.calcHist([frame], [0, 1], None, [180, 256], [0, 180, 0, 256])
                # cv2.imshow("2D Histogram", hist)

                # create a colored histogram from frame and shot a 2d histogram plot
                # siehe https://blog.helmutkarger.de/raspberry-video-camera-teil-20-exkurs-farbdarstellung-per-2d-histogramm/
                hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                      # convert to HSV
                hist2 = cv2.calcHist([hsv2],[0,1],None,[30,30],[0,180,0,256])     # calc hist satu + hue
                # hist2[0,0]=0   nur bei freigestelltem Bild                                                  # delete bachground
                cv2.normalize(hist2,hist2,0,100,cv2.NORM_MINMAX)                  # normalize to 100
                try:
                    plt.figure(1)
                # set interactive mode on
                    plt.ion()
                    # plt.close()
                    plt.clf()
                    plt.imshow(hist2.T, interpolation = "nearest", origin = "lower") # plot histogram
                    plt.title('Hue - Saturation - Histogram')
                    plt.xlabel('Hue')
                    plt.ylabel('Saturation')
                    plt.colorbar(ticks=(0,20,40,60,80,100))
                    plt.show()

                except Exception as ex:
                    print("error: " + str(ex))

        #--------------------------------------------------------------------------------------------
        # find color class of the iron bead in the captured image
        # >>>>>>>>>>> COLOR CLASS DETECTION <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<       
        # colorClass, highestSimilarity = findColorClassUsingHistograms(frame, mask, trainingImageHistograms)
        # lastMeasureTimestamp = time.time()
        # only search a color, if circles were found 
        colorClass = ""
        if x > 75 and y > 0:
            # show the frame now with the mask
            # apply mask to frame
            frame = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow("Frame", frame)

            # repeat three times to find circles and find color class using histograms
            # analyze, if the color class is the same in all three measurements
            colorClassMeasurements = []
            
            for i in range(3):
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")
                highestSimilarity = 0

                (grabbed, frame) = camera.read()
                # resize the frame
                frame = imutils.resize(frame, width=800)
                # show the frame
                # find circles in the image
                frame, x, y, mask = findCircleDuringSorting(frame)	
                # apply mask to frame
                frame = cv2.bitwise_and(frame, frame, mask=mask)
                cv2.imshow("Frame", frame)

                if (0==1):
                    # find circles in frame 
                    frame_circle, x, y, mask = findCircleDuringSorting(frame)
                    # apply mask to frame_circle
                    frame_circle = cv2.bitwise_and(frame_circle, frame_circle, mask=mask)
                    # show the frame
                    cv2.imshow("FrameCircle"+str(i), frame_circle)

                # create a black circle mask with center x,y and radius 200
                if x > 75 and y > 0:
                    colorClass, highestSimilarity = findColorClassUsingHistograms(frame, mask, trainingImageHistograms)
                    # append the color class and highest similarity to the list
                    colorClassMeasurements.append([colorClass, highestSimilarity])
                        # if any similarity is above 0.8, the color class is from this measurement
                if highestSimilarity > 0.75:
                    # no further measurements needed
                    break

            if len(colorClassMeasurements) >= 2: 
                # analyze, if a color class was found more than once 
                # extract the color classes from colorClassMeasurements
                highestSimilarityVerified = 0
                colorClassesinMeasurement = []
                colorClassToVerify = ""
                for i in range(len(colorClassMeasurements)):
                    colorClassesinMeasurement.append(colorClassMeasurements[i][0])
                # count the color classes in colorClassMeasurements using Counter()
                # Counter() returns a dictionary with the color classes as keys and the number of occurences as values
                # example: Counter({'red': 2, 'green': 1})
                measurementSummary = dict(Counter(colorClassesinMeasurement))
                # sort measurementSummary by value descending
                measurementSummary = sorted(measurementSummary.items(), key=lambda x: x[1], reverse=True)
                # if the first line in measurementSummary has a value of 2 average the highetsSimilarity of the two measurements
                # with the same color class as in line 1 of measurementSummary
                if measurementSummary[0][1] == 2:
                    # get the color class from the first line in measurementSummary
                    colorClassToVerify = measurementSummary[0][0]
                    # loop at the colorClassMeasurements list and average the higestSimilarity of the two measurements
                    # with the same color class as in line 1 of measurementSummary
                    for i in range(len(colorClassMeasurements)):
                        if colorClassMeasurements[i][0] == colorClassToVerify:
                            if highestSimilarityVerified == 0:
                                highestSimilarityVerified = colorClassMeasurements[i][1]
                            else:
                                highestSimilarityVerified = (highestSimilarityVerified + colorClassMeasurements[i][1]) / 2
                        
            if len(colorClassMeasurements) == 1 and colorClassMeasurements[0][1] > 0.75: 
                colorClass = colorClassMeasurements[0][0]
                averageSimilarity = colorClassMeasurements[0][1]
            elif len(colorClassMeasurements) == 2 and \
                    (colorClassMeasurements[1][1] > 0.75 or \
                     colorClassMeasurements[0][1] > 0.75): 
                colorClass = colorClassMeasurements[1][0]
                averageSimilarity = colorClassMeasurements[1][1]
            elif len(colorClassMeasurements) == 3 and \
                        (colorClassMeasurements[2][1] > 0.75 or \
                         colorClassMeasurements[1][1] > 0.75 or \
                         colorClassMeasurements[0][1] > 0.75): 
                colorClass = colorClassMeasurements[2][0]
                averageSimilarity = colorClassMeasurements[2][1]
            # analyze, if the color class is the same in all three measurements
            # if colorClassMeasurements has three entries
            elif ( len(colorClassMeasurements) == 3 and
               (colorClassMeasurements[0][0] == colorClassMeasurements[1][0] and 
                colorClassMeasurements[0][0] == colorClassMeasurements[2][0])):
                # calculate the average similarity
                averageSimilarity = (colorClassMeasurements[0][1] + 
                                     colorClassMeasurements[1][1] + 
                                     colorClassMeasurements[2][1]) / 3
                if averageSimilarity > 0.1:
                    colorClass = colorClassMeasurements[0][0]
            elif colorClassToVerify != "" and highestSimilarityVerified > 0.5:
                colorClass = colorClassToVerify
                averageSimilarity = highestSimilarityVerified
            else: 
                colorClass = "empty"
                averageSimilarity = 1


                                  
            # ----- add additional info in the captured image -------------------------------------------
            CapturedFrame = frame.copy()  # remember original frame before adding info
            # add the color class on the captured image
            cv2.putText(frame, colorClass, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # add the similarity on the captured image
            # similarity text shall have a red color if the similarity is too low
            if averageSimilarity < 0.4:
                textcolor = (0, 0, 255)
            elif averageSimilarity < 0.7:
                # text color orange
                textcolor = (0, 165, 255)
            else:
                # text color green
                textcolor = (0, 255, 0)        
            cv2.putText(frame, str(averageSimilarity), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, textcolor, 1)
            
            # add the found color classes and similarities from the last three measurements on the captured image
            # if colorClassMeasurements has three entries
            # if len(colorClassMeasurements) == 3:

            try: # let it fail, if colorClassMeasurements has not three entries
                if len(colorClassMeasurements) >= 1:
                    cv2.putText(frame, colorClassMeasurements[0][0], (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.putText(frame, colorClassMeasurements[1][0], (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                if len(colorClassMeasurements) >= 2:
                    cv2.putText(frame, colorClassMeasurements[2][0], (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.putText(frame, str(colorClassMeasurements[0][1]), (100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                if len(colorClassMeasurements) >= 3:
                    cv2.putText(frame, str(colorClassMeasurements[1][1]), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.putText(frame, str(colorClassMeasurements[2][1]), (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            except Exception as ex: 
                # do nothing
                print("error measurement display: " + str(ex))

            # show the frame
            cv2.imshow("Frame", frame)

            #----- move to the bin for the found color class ----------------------------------------------
            # if colorClass is not lastMeasuredColorClass:
            # move to the bin for the found color class
            if colorClass != lastMeasuredColorClass:  
                lastMeasuredColorClass = colorClass
                # move to the bin for the found color class; turn off to make training pictures
                if ( not training): 
                    wait_for_arduino_ready4cmd_withtimeout(ser) 
                    
                    # write to serial port non blocking
                    try:
                        if averageSimilarity < 0.1:
                            # move to bin 0 
                            ser.write(b'b0\n')
                            # play a beep sound
                            # winsound.Beep(1000, 1000)

                        else:
                            send_command_with_bin_number(ser, colorClass)
                    except Exception as ex:
                        # print error to console in red color
                        print("\033[91m" + "error: " + str(ex) + "\033[0m")

            # move isolator wheel to eject the bead and avoid a further measurement of the same bead        
            wait_for_arduino_ready4cmd_withtimeout(ser) 
            ser.write(b'm\n') 
            # wait 200ms
            cv2.waitKey (700) 
            #--------------------------------------------------------------------------------------------
            # ----- remember last found not empty color class as basis to detect the next bead -----
            # store last found not empty color class
            if colorClass != 'empty':
                lastFoundFrame = CapturedFrame.copy()
                lastFoundFrame = imutils.resize(lastFoundFrame, width=200) 
                lastFoundColorClass = colorClass
                lastSimilarity = averageSimilarity

                # add the last found not empty color class on the captured image
                cv2.putText(lastFoundFrame, "L:"+lastFoundColorClass, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                # add the last found not empty similarity on the captured image
                cv2.putText(lastFoundFrame, str(lastSimilarity), (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # show the frame
                cv2.imshow("Last Found Frame", lastFoundFrame)       
                       
            #--------------------------------------------------------------------------------------------
            # create files
            # if the similarity to any existing training picture is too low, create new training picture with prefix from input
            # if the similarity to any existing training picture is high enough, create new classified picture with prefix from input
            #    
            # manual procedure: check the classified picture for wrong classification and add such pictures to the training pictures
            #    to create a series of empty pictures, use the prefix 'empty' and adapt the following coding
            #    for continous generation of 'empty' training files
            if training:   # turn on to take training pictures
                # the first part of the if makes sure that for all found color classes a training picture is created
                if  colorClass != "empty" or highestSimilarity < 0.3:
                    # key = ord("s") # for "empty" images
                    # wait 200ms
                    #cv2.waitKey (200)
                    # if key = s, save the image to a file
                    # if key == ord("s"):
                    # create a filename
                    filename = "C:\\Users\\robert\\Documents\\pythonRobert\\IronBeadsColorDetect\\trainingsImages\\" + filenamePrefix + str(filenameCounter) + ".png"
                    # save the image to the file
                    cv2.imwrite(filename, CapturedFrame)
                    # flush the image to the file

                    # increment the filename counter
                    filenameCounter += 1

                    # flush  file
                                
                    #wait 400ms
                    #cv2.waitKey (200)
                elif colorClass != 'empty' and highestSimilarity > 0.4:
                    # store image as correctly classified image
                    filename = "C:\\Users\\robert\\Documents\\pythonRobert\\IronBeadsColorDetect\\ClassifiedImages\\" + colorClass + str(filenameCounter) + "_withInfo.png"
                    # save the image to the file
                    # cv2.imwrite(filename, frame)
                    # create a filename
                    filename = "C:\\Users\\robert\\Documents\\pythonRobert\\IronBeadsColorDetect\\ClassifiedImages\\" + colorClass + str(filenameCounter) + ".png"
                    # save the image to the file
                    # cv2.imwrite(filename, CapturedFrame)
                    # flush the image to the file
                    filenameCounter += 1
                elif colorClass != 'empty':
                    no_unclassified += 1
                    # write no unclassified to console  in red color
                    print("\033[91m" + "no unclassified: " + str(no_unclassified) + 'colorClass:' + colorClass + "\033[0m")  

        else: # no circles found
            ser.write(b'g\n')  # go on, turn the wheel   
       
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
    ser.close()	
    sys.exit()
