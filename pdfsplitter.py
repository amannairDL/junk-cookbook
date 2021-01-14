# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:40:20 2020

@author: Aman.Sivaprasad
"""

import os
from PyPDF2 import PdfFileReader, PdfFileWriter

def pdf_splitter(path):
    fname = os.path.splitext(os.path.basename(path))[0]
    pdf = PdfFileReader(path)
    for page in range(pdf.getNumPages()):
        pdf_writer = PdfFileWriter()
        pdf_writer.addPage(pdf.getPage(page))
        output_filename = '{}_page_{}.pdf'.format(
            fname, page+1)
        
        with open(r"C:\Users\Aman.Sivaprasad\Desktop\labcorp\04032020\testing_newfiles\700_800_3200_color_600dpi\testing_checkbox2"+"\\"+output_filename, 'wb') as out:
            pdf_writer.write(out)
        print('Created: {}'.format(output_filename))



if __name__ == '__main__':
    path = r"C:\Users\Aman.Sivaprasad\Desktop\labcorp\04032020\testing_newfiles\700_800_3200_color_600dpi\testing_checkbox2\Testing Sample.pdf"
    pdf_splitter(path)