# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:53:40 2020

@author: Aman.Sivaprasad
"""

from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

file_path5 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\page_3.jpg"

file_path7 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\TrainingData\800_____20200427 Training Data_Page_7.jpg"
file_path8 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\TrainingData\800_____20200427 Training Data_Page_8.jpg"
file_path9 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\TrainingData\800_____20200427 Training Data_Page_9.jpg"
file_path10 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\TrainingData\800_____20200427 Training Data_Page_10.jpg"
file_path1 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\TrainingData\800_____20200427 Training Data_Page_1.jpg"
file_path2 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\TrainingData\800_____20200427 Training Data_Page_2.jpg"



image_to_rotate = cv2.imread(file_path1)


#rotation angle in degree
rotated = ndimage.rotate(image_to_rotate, -20)


plt.imshow(rotated)

output_path = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\TrainingData\ori_angle"+"800_____20200427 Training Data_Page_1_20.jpg"
cv2.imwrite(output_path,rotated)





