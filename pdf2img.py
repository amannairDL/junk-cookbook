# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:22:02 2020

@author: Aman.Sivaprasad
"""

from PIL import Image 
import sys 
from pdf2image import convert_from_path 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 



output_filepath = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\forcasting\rahulc\OneDrive_1_10-29-2020\firstdrive"
folder_path = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\forcasting\rahulc\OneDrive_1_10-29-2020"
joiner = "\\"


for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        PDF_file = folder_path + joiner +filename
        print(PDF_file)
        pages = convert_from_path(PDF_file, 500)
        file_name = filename.rsplit('.', 1)[0] + '.jpg' #filename[:-4] + ".jpg"
        print(file_name)
        output_file = output_filepath+joiner+ file_name
        pages[0].save(output_file, 'JPEG')
        print(output_file)
        

        