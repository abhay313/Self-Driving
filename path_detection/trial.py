import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)
    return img

def process(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [(0, height),(width/6,height/2.5),(width/1.4, height/2.5),(width, height)]
    image = cv2.equalizeHist(image)
    canny_image = cv2.Canny(image, 100, 120)
    cropped_image = region_of_interest(canny_image,np.array([region_of_interest_vertices], np.int32))
    return cropped_image

first_frame = None
kernel = np.ones((5,5), np.uint8)
cap = cv2.VideoCapture('test.mp4')

while True:
    check, frame = cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if first_frame is None :
        first_frame=gray
        continue
    cv2.imshow('gaussian', gray)
    cropped_td = process(gray)
    cv2.imshow('roi', cropped_td)
    key = cv2.waitKey(1)
    if key == ord('q') :
        break
        
cap.release()
cv2.destroyAllWindows()
