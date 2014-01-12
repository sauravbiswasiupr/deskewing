#!/usr/bin/python 

from numpy import *
from pylab import *
from matplotlib import pyplot as plt
import cPickle 



f=open('../datasets/train_26.pkl','rb')
train_x,train_y=cPickle.load(f)

print "Selecting a sample image and its skewed version..."
maxwidth=29 ; maxheight=32

sample_image=train_y[120].reshape((maxwidth,maxheight))
skewed_image=train_x[120].reshape((maxwidth,maxheight))

print "binarizing the skewed image ...." 
c=1*(skewed_image !=0 )
d=1*(sample_image != 0)

figure()
subplot(121)
plt.imshow(c,cmap='gray')
subplot(122)
plt.imshow(d,cmap='gray')
show()
