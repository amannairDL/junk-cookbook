# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:10:57 2020

@author: Aman.Sivaprasad
"""


import json

with open(r"C:\Users\Aman.Sivaprasad\Desktop\newtest.json", ) as f:
  data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print(data)

filepath=r"C:\Users\Aman.Sivaprasad\Desktop\akshay\1- ATEA(Norway) to ATEA UAB (Lithuania)\1. ATEA(Norway) to ATEA UAB (Lithuania)\Result-Year 2 - 902673245.json"
with open(filepath) as f:
          data = json.load(f)
          formatted_json_str1 = json.dumps(data, indent = 4,sort_keys=True)


text_file = open(r"C:\Users\Aman.Sivaprasad\Desktop\akshay\1- ATEA(Norway) to ATEA UAB (Lithuania)"+"\\"+"Output.txt", "w")
text_file.write(formatted_json_str1)
text_file.close()




with open(filepath, 'r') as j:
     contents = json.loads(j.read())
     
     
with open("Aaman.json",encoding='utf-16', errors='ignore') as json_data:
     data = json.load(json_data, strict=False)
     


import pprint
formatted_json_str = pprint.pformat(data)
print(formatted_json_str)