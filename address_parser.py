# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:08:26 2020

@author: Aman.Sivaprasad
"""

from addressnet.predict import predict_one


text = "casa del gelato, 10A 24-26 high street road mount waverley vic 3183"
text2 = "Van Siclen Avenue and Flatlands Avenue Brooklyn, New York 11207"
text3 = "Tower 535 - 11007(4), 535 Jaffe Road, Causeway Bay HK, China"
text4 = "Suite 803, 55 Wall Street New York, USA"
text5 = 'The Book Club 100-106 Leonard St, Shoreditch, London, Greater London, EC2A 4RH, United Kingdom'
text6 = '123 West Mifflin Street, Madison, WI, 53703'

out = predict_one(text)
out2 = predict_one(text2)
out3 = predict_one(text3)
out4 = predict_one(text4)
out5 = predict_one(text5)

print(out)
print(out2)
print(out3)
print(out4)



#using docker
#
#$ git clone https://github.com/anandaroop/try-postal.git
#$ cd try-postal
#$ docker build -t try-postal .
#$ docker container run -it -p 5000:5000 try-postal



C:\Users\Aman.Sivaprasad\Music>docker run -it --rm -p 8888:8888 -v C:\Users\Aman.Sivaprasad\Music:/home/jovyan/work riordan/docker-jupyter-scipy-notebook-libpostal


from postal.parser import parse_address
parse_address()



#address
#addressnet # austrialia
#usaddress #usa
#pypostal #alll
#postal-address #European

#googleapi #https://github.com/thilohuellmann/address_transformation/blob/master/address_transformation.py

#https://www.kaggle.com/stefanjaro/libpostal-windows-and-jupyter-notebook

from postal.parser import parse_address
parse_address('The Book Club 100-106 Leonard St Shoreditch London EC2A 4RH, United Kingdom')



from address import AddressParser, Address

ap = AddressParser()
address = ap.parse_address(text3)
print(address)
print("Address is: {0} {1} {2} {3}".format(address.house_number, address.street_prefix, address.street, address.street_suffix))

#Address is: 123 W. Mifflin St.