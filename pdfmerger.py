# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:47:16 2020

@author: Aman.Sivaprasad
"""

from PyPDF2 import PdfFileMerger

pdfs = ['aman 10th and 12th.pdf', 'aman_srm marksheets (1).pdf']

merger = PdfFileMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write(r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\learnml\collect\MARKSHEETS.pdf")
merger.close()