# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (33,80,40)
greenUpper = (102,255,255)
pts = deque(maxlen=args["buffer"])
 
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
 
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
 
# allow the camera or video file to warm up
time.sleep(2.0)

rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/training_data.yml")

# keep looping
while True:
	# grab the current frame
	frame = vs.read()
 
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
 
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
 
	# resize the frame, blur it, and convert it to the HSV
	# color space	
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
 
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #left
	lineThickness = 2
	#---------------atas--------bawah-panjang-----------------------
	cv2.line(frame, (320, 180), (320, 320), (0,255,0), lineThickness)
        #right
	lineThickness = 2
	cv2.line(frame, (550, 200), (550, 340), (0,255,0), lineThickness)
	#up
	cv2.line(frame, (320, 180), (550, 200), (0,255,0), lineThickness)
	#down
	cv2.line(frame, (550, 340), (320, 320), (0,255,0), lineThickness)
	#right 2
	cv2.line(frame, (580, 220), (580, 330), (255,255,255), lineThickness)
	#up right
	cv2.line(frame, (550, 200), (580, 220), (255,255,255), lineThickness)
	#down right
	cv2.line(frame, (550, 340), (580, 330), (255,255,255), lineThickness)
	#left 2
	cv2.line(frame, (350, 200), (350, 320), (255,255,255), lineThickness)
	#up left
	cv2.line(frame, (320, 180), (350, 200), (255,255,255), lineThickness)
	#down left
	cv2.line(frame, (320, 320), (350, 320), (255,255,255), lineThickness)
	#up 2
	cv2.line(frame, (350, 200), (580, 220), (255,255,255), lineThickness)


	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		text2="GOAL"

		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			

		for i in range(len(cnts)):x,y,w,h=cv2.boundingRect(cnts[i])
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
		print ("X : "+str(x),"Y : "+str(y))
		id,conf=rec.predict(gray[y:y+h,x:x+w])
		#if(id==1) & (y > 195 & y < 250 & x > 325 & x < 470):text2="GOAL"
		if y > 195 & y < 250 & x > 325 & x < 470:text2="GOAL"
		if(id==1):text="BOLA"
		else:text="UNKNOWN"
		cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
		cv2.putText(frame, text2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
						
		
	# update the points queue
	pts.appendleft(center)

	# show the frame to our screen
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q") :
		break
 
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
 
# otherwise, release the camera
else:
	vs.release()
 
# close all windows
cv2.destroyAllWindows()
