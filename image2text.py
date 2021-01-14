# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:49:09 2020

@author: Aman.Sivaprasad
"""

from PIL import Image 
import pytesseract 
import sys 
from pdf2image import convert_from_path 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 
pytesseract.pytesseract.tesseract_cmd=r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Documents\Tesseract-OCR\tesseract.exe" 

# https://nanonets.com/blog/ocr-with-tesseract/
# https://stackabuse.com/pytesseract-simple-python-optical-character-recognition/
# https://docparser.com/blog/improve-ocr-accuracy/
# https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
#https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/






# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated













# Path of the pdf 
PDF_file = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\demo\input\data\700_form_702_800_page_10.pdf"

# Store all the pages of the PDF in a variable 
pages = convert_from_path(PDF_file, 500) 

# Counter to store images of each page of PDF to image 
image_counter = 1
filepath = r"C:\Users\Aman.Sivaprasad\Desktop\pocHRresume\invoce_data\Invoice Copies\18122019\firstcheck"
# Iterate through all the pages stored above 
for page in pages: 

	filename = "page_mercantile"+str(image_counter)+".jpg"
	
	# Save the image of the page in system 
	page.save(filepath+"\\"+ filename, 'JPEG') 

	# Increment the counter to update filename 
	image_counter = image_counter + 1















## pre-processing image 

#filename = filepath+"\\"+ filename
filename = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\invoice acme\dataset\singlepage_pdf\training_data\non_skew\image\table1.jpg"

img = cv2.imread(filename)
# Convert to gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply dilation and erosion to remove some noise
kernel = np.ones((5,5),np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)
# Apply blur to smooth out the edges
img = cv2.GaussianBlur(img, (5, 5), 0)

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img = cv2.filter2D(img, -1, kernel)

# Apply threshold to get image with only b&w (binarization)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


plt.figure(figsize = (5*9, 5*10))
plt.imshow(img ,cmap="CMRmap")
#'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r'










#===================img preporcessing=================================
#removing vertical lines
#testing processing
# https://stackoverflow.com/questions/33949831/whats-the-way-to-remove-all-lines-and-borders-in-imagekeep-texts-programmatic

image = cv2.imread(filename)


kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
temp1 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_vertical)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
temp2 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, horizontal_kernel)
temp3 = cv2.add(temp1, temp2)
result = cv2.add(temp3, image)


plt.figure(figsize = (5*9, 5*10))
plt.imshow(result ,cmap="CMRmap")








#===================img preporcessing=================================
#removing horizontal lines

result = img.copy()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


# Remove horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(result, [c], -1, (255,255,255), 5)




# Remove vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(result, [c], -1, (255,255,255), 5)




plt.figure(figsize = (5*9, 5*10))
plt.imshow(result ,cmap="CMRmap")



#========================================================================












#testing
#oem
#0    Legacy engine only.
#1    Neural nets LSTM engine only.
#2    Legacy + LSTM engines.
#3    Default, based on what is available.

#psm
#Page segmentation modes:
#  0    Orientation and script detection (OSD) only.
#  1    Automatic page segmentation with OSD.
#  2    Automatic page segmentation, but no OSD, or OCR.
#  3    Fully automatic page segmentation, but no OSD. (Default)
#  4    Assume a single column of text of variable sizes.
#  5    Assume a single uniform block of vertically aligned text.
#  6    Assume a single uniform block of text.
#  7    Treat the image as a single text line.
#  8    Treat the image as a single word.
#  9    Treat the image as a single word in a circle.
# 10    Treat the image as a single character.
# 11    Sparse text. Find as much text as possible in no particular order.
# 12    Sparse text with OSD.
# 13    Raw line. Treat the image as a single text line,
#                        bypassing hacks that are Tesseract-specific.




custom_config = r'--oem 3 -l eng --psm 6'
text = str(((pytesseract.image_to_string(result , lang="eng" , config=custom_config)))) 












# Creating a text file to write the output 
outfile = r"C:\Users\Aman.Sivaprasad\Desktop\poctesting2\pdf\IP Release CDP Checklist\OCR-IP Release CDP Checklist - 12-Jun-2019.txt"

# Open the file in append mode so that 
# All contents of all images are added to the same file 
f = open(outfile, "a") 

 

# page_n.jpg 

#increasing the acc
# Adding custom options
custom_config = r'--oem 3 -l eng --psm 6'
# Recognize the text as string in image using pytesserct 
#text = str(((pytesseract.image_to_string(img , lang="eng" , config=custom_config)))) 

text = str(((pytesseract.image_to_string(img , config=custom_config)))) 


text = text.replace('-\n', '')	 

# Finally, write the processed text to the file. 
f.write(text) 

# Close the file after writing all the text. 
f.close()











