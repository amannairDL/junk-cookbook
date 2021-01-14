# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:08:18 2020

@author: Aman.Sivaprasad
"""

import img2pdf 
from PIL import Image 
import os 
  
# storing image path 
img_path = "C:/Users/Admin/Desktop/GfG_images/do_nawab.png"
  
# storing pdf path 
pdf_path = "C:/Users/Admin/Desktop/GfG_images/file.pdf"
  
# opening image 
image = Image.open(img_path) 
  
# converting into chunks using img2pdf 
pdf_bytes = img2pdf.convert(image.filename) 
  
# opening or creating pdf file 
file = open(pdf_path, "wb") 
  
# writing pdf files with chunks 
file.write(pdf_bytes) 
  
# closing image file 
image.close() 
  
# closing pdf file 
file.close() 
  
# output 
print("Successfully made pdf file") 






 1 import img2pdf
 2 import argparse
 3 
 4 def process_images(min_range, max_range, prefix, suffix, out_file):
 5     images = []
 6     for i in range(min_range, max_range + 1):
 7         fname = prefix + str(i) + suffix
 8         images.append(fname)
 9     out_file.write(img2pdf.convert(images))
10 
11 if __name__ == "__main__":
12     # Let the user pass parameters to the code, all parameters are optional have some default values
13     # ...
14 
15     # Make sure the output file ends with *.pdf*
16     # ...
17 
     with open(out_fname, "wb") as out_file:
         process_images(min_range, max_range, prefix, suffix, out_file)




import os
import img2pdf
with open(r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\dob_hypen\output.pdf", "wb") as f:
    f.write(img2pdf.convert(([i for i in os.listdir(r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\dob_hypen") if i.endswith(".jpg")]))


image_data = open(image_file_name, "rb").read()


page_1.jpg



img = cv2.imread(r"C:/Users/Aman.Sivaprasad/OneDrive - EY/Desktop/labcorp/results/testingdata/dob/ddob_hypen/page_1.jpg",0)


from PIL import Image  








import os
import img2pdf
import cv2


ab_imagedata=[]
ab = [i for i in os.listdir(r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\ddob_hyphen") if i.endswith(".jpg")]


for i in ab:
    path = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\ddob_hypen"+"\\"+i
    print(path)
    img = cv2.imread(path)
    img.shape
#    width = 4250
#    height = 5500
#    dim = (width, height)
#    img = cv2.resize(img, dim)
#    cv2.imwrite(path, img)


for i in ab:
    path = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\ddob_hypen"+"\\"+i
    print(path)
    image_data = open(path, "rb").read()
    ab_imagedata.append(image_data)
        
with open(r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\ddob_hypen\hyphen_new_1.pdf", "wb") as f:
	f.write(img2pdf.convert(ab_imagedata))

