import matplotlib.pylab as plt
import cv2
import numpy as np


def region_of_interest(image,vertices):
	mask=np.zeros_like(image)
	match_mask_color= 255
	cv2.fillPoly(mask,vertices,match_mask_color)
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image

img =cv2.imread('road.png')

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


print(img.shape)

height=img.shape[0]
width=img.shape[1]

region_of_interest_vertices=[(0, height),(width/2, height/2),(width, height)]

gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
canny_img=cv2.Canny(gray_img,50,250)
masked_img = region_of_interest(canny_img,np.array([region_of_interest_vertices],np.int32),)


lines=cv2.HoughLinesP(masked_img,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

for line in lines:
	x1,y1,x2,y2=line[0]
	cv2.line(img,(x1,y1),(x2,y2),(255,0,0),4)

cv2.imshow('LINES',masked_img)
plt.imshow(img)


plt.show()
