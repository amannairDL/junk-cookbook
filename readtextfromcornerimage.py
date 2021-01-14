# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:38:00 2020

@author: Aman.Sivaprasad
"""


import cv2
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd=r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Documents\Tesseract-OCR\tesseract.exe" 


img = cv2.imread(r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\page_1.jpg")

 
x2 = img.shape[1]
x1 = x2 - 400
y1 = 0
y2 = 500


img1 = img.copy()

cv2.rectangle(img1,(x1,y1),(x2,y2),(0,255,0),3)
crop_img1 = img[y1:y2, x1:x2]


plt.figure(2, figsize = (5*9, 5*10))
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))


plt.imshow(cv2.cvtColor(crop_img1, cv2.COLOR_BGR2RGB))



text = str(((pytesseract.image_to_string(crop_img1))))
