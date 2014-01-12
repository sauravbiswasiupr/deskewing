#!/usr/bin/python 
from os import system 
skews=[-3,-2,-1,0,1,2,3]
for skew in skews:
   system('python create_dataset.py '+str(skew))
print "Data created successfully..."
