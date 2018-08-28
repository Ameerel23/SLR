# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:43:00 2018

@author: xinwe
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

#import serial
#import time
#import matplotlib.pyplot as mp
left = 0
img_counter = 0
count = 0
b = open('XYAxisTake10.csv', 'a')
a = csv.writer(b)
 
#action = 'm'
#currentAction = 'm'
#statePrevious = None

# Connect to Arduino's serial port
#serial = serial.Serial("/dev/ttyUSB0", 19200)

vid = cv2.VideoCapture(1)

while True:
    count += 1
    ret,frame = vid.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, (10, 100, 20), (25, 255, 255))
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask and initialize the current
	# (x, y) center of the ball
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    
    # Proceed only if at least one contour was found
    if len(cnts) > 0:
        # Find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max (cnts, key = cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)

       
        
        
        #this will change according to the camera
        if x == 240.0:
            # whenever y value reach 240, it will snap a picture
            print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
            img_name = "frame_{}.jpg".format(img_counter)
            image = cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            image = cv2.imread("opencv_frame_{}.jpg".format(img_counter))
            img_counter += 1
        
        if x == 500.0:
            # whenever y value reach 240, it will snap a picture
            print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
            img_name = "frame_{}.jpg".format(img_counter)
            image = cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            image = cv2.imread("opencv_frame_{}.jpg".format(img_counter))
            img_counter += 1
            
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if radius < 25:
            # Draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)
            cv2.putText(frame,"centroid", (center[0]+10,center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
            cv2.putText(frame,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
            print (x,y)
            if y > 241:
                left = 0
            elif y < 240:
                left = 1
            data = [[y,left]]
            a.writerows(data)
            b.close            

            
#    else: 
#        result  = "No Ball Detected"
#        action = 'm'
#        if result != statePrevious:
#            time.sleep(2)
#    if currentAction != action:
#        #serial.write(action)
#        currentAction = action
#    statePrevious = result 
            
            
    
    
    # show the frame to our screen
    cv2.imshow("Top View Camera", frame)
    #	cv2.imshow("Thresh", thresh)
    #	cv2.imshow("Mask", mask)
    if count == 1:
        cv2.moveWindow("Top View Camera", 1115, 220)	
        #cv2.moveWindow("Thresh", 710, 10)
        #cv2.moveWindow("Mask", 1410, 10)
    
    k= cv2.waitKey(1)
    
    if k%256 == 27:
        break
    
# Close serial port + Release video capture
#serial.write('m')
#serial.close()
vid.release()
cv2.destroyAllWindows()
print ("Program Exited Successfully")

dataset = pd.read_csv('XYAxisTake10.csv')
pltAll = dataset.iloc[:].values
Y = dataset.iloc[:, :1].values
l = dataset.iloc[:,-1].values
            
from sklearn.cross_validation import train_test_split
Y_train, Y_test, l_train, l_test = train_test_split(Y, l, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(Y_train, l_train)

# Predicting the Test set results
l_pred = clf.predict(Y_test)

# Visualising the Training set results
plt.scatter(Y_train, l_train, color = 'red')
plt.plot(Y_train, clf.predict(Y_train), color = 'blue')
plt.title('Ball Location (Training set)')
plt.xlabel('Y axis point')
plt.ylabel('Left or Right')
plt.show
            
plt.scatter(Y_test, l_test, color = 'red')
plt.plot(Y_train, clf.predict(Y_train), color = 'blue')
plt.title('Ball Location (Test set)')
plt.xlabel('Y axis point')
plt.ylabel('Left or Right')
plt.show