import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture("car2.mp4")
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print(frames_count, fps, width, height)

# create background subtractor
sub = cv2.createBackgroundSubtractorMOG2()  

# information to start saving a video file
# import image
ret, frame = cap.read()
# resize ratio
ratio = 1.0 
# resize image
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  
width2, height2, channels = image.shape
#video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)

while True:
    # import image
    ret, frame = cap.read()
    #if video finish repeat
    if not ret: 
        frame = cv2.VideoCapture("car2.mp4")
        break
    # if there is a frame continue with code
    if ret: 
        # resize image
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  
        # converts image to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        # uses the background subtraction
        fgmask = sub.apply(gray) 
     
        # applies different thresholds to fgmask to try and isolate cars
        # just have to keep playing around with settings until cars are easily identifiable
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("closing", closing) 
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  
        dilation = cv2.dilate(opening, kernel)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
        # creates contours
        im2, contours, hierarchy = cv2.findContours(bins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minarea = 400
        # max area for contours, can be quite large for buses
        maxarea = 50000
        # vectors for the x and y locations of contour centroids in current frame
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))
        
        # cycles through all contours in current frame
        for i in range(len(contours)): 
            # using hierarchy to only count parent contours (contours not within others)
            if hierarchy[0, i, 3] == -1:  
                # area of contour
                area = cv2.contourArea(contours[i])  
                 # area threshold for contour
                if minarea < area < maxarea: 
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # gets bounding points of contour to create rectangle
                    # x,y is top left corner and w,h is width and height
                    x, y, w, h = cv2.boundingRect(cnt)
                    # creates a rectangle around contour
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Prints centroid text in order to double check later on
                    cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
                    cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
                    
    cv2.imshow("countours", image)
    key = cv2.waitKey(20)  
    if key == 27:
       break
    
cap.release()
cv2.destroyAllWindows() 
