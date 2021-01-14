# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:21:32 2020

@author: Aman.Sivaprasad
"""

# Import required packages 
import cv2 
import pytesseract 
import matplotlib.pyplot as plt
  
# Mention the installed location of Tesseract-OCR in your system 
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Documents\Tesseract-OCR\tesseract.exe" 
 

#text bounding box
 
# Read image from which text needs to be extracted 
img = cv2.imread(r"C:\Users\Aman.Sivaprasad\Downloads\sample4.jpg") 
 
# Preprocessing the image starts 
img = result.copy()
# Convert the image to gray scale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 

# Performing OTSU threshold 
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
  
# Specify structure shape and kernel size.  
# Kernel size increases or decreases the area  
# of the rectangle to be detected. 
# A smaller value like (10, 10) will detect  
# each word instead of a sentence. 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) 
  
# Appplying dilation on the threshold image 
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
  
# Finding contours 
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE) 
  
# Creating a copy of image 
im2 = img.copy() 







#===================img preporcessing=================================
#getting cleaner image removing noise




import cv2
import numpy as np

image = cv2.imread('1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
result = 255 - close

cv2.imshow('sharpen', sharpen)
cv2.imshow('thresh', thresh)
cv2.imshow('close', close)
cv2.imshow('result', result)
cv2.waitKey()



#==================================================================









  
# A text file is created and flushed 
file = open(r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\invoice acme\dataset\singlepage_pdf\training_data\non_skew\image\recognized.txt", "w+") 
file.write("") 
file.close() 
  
# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 

# Open the file in append mode 
res = ""
for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt) 
      
    # Drawing a rectangle on copied image 
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      
    # Cropping the text block for giving input to OCR 
    cropped = im2[y:y + h, x:x + w] 
      
    
      
#    # Apply OCR on the cropped image 
#    text = pytesseract.image_to_string(cropped) 
#    
#    z = x + w
#    v = y + h
#    nextw = str(z) + " , " + str(v) + ") "
#    # Appending the text into file 
#    value = text + " (" + str(x) +"," + str (y) + ") " + " (" + nextw
#    value = str(value)
#    res = res + value
#    res = res + "\n"
      
    # Close the file 

plt.figure(figsize = (5*9, 5*10))
plt.imshow(im2 ,cmap="CMRmap")

file = open(r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\invoice acme\dataset\singlepage_pdf\training_data\non_skew\image\recognized.txt", "a") 
file.write(res)
file.close()
   
cv2.imwrite(r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\invoice acme\dataset\singlepage_pdf\training_data\non_skew\image\cleantable1.jpg" ,im2)























