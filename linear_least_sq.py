'''Script to see if we can find the transformation matrix for deskewing skewed images. The transformation matrix W is found by using a linear least square technique'''
from numpy import *
from pylab import *
from matplotlib import pyplot as plt
from numpy.linalg import pinv,norm
import cPickle 
import argparse


parser=argparse.ArgumentParser()
parser.add_argument("skewangle",type=int,help="The skew which needs to be learnt by the linear least square method")
parser.add_argument("--verbose",help="verbosity turned on",action="store_true")
parser.add_argument("--binarize",help="binarize the images", action="store_true")

args=parser.parse_args()
angle=args.skewangle
if args.verbose:
   print "Module that learns deskewing on skewed images by using a linear least square approach"

f=open('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/datasets/train_'+str(angle)+'.pkl','rb') 
train_x,train_y=cPickle.load(f)
f.close()


f=open('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/datasets/test_'+str(angle)+'.pkl','rb') 
test_x,test_y=cPickle.load(f) 
f.close() 


print "Dataset loaded..."
print "Finding the transformation matrix using linear least square approach ..."

assert len(train_x)==2340
assert len(test_x)==260
if args.binarize:
  train_x=1*(train_x != 0)
  train_y=1*(train_y != 0)
  test_x=1*(test_x != 0)
  test_y=1*(test_y != 0)

print "----------TRAIN SET-------------------"
print amax(train_x) , amin(train_x) 
print amax(train_y) , amin(train_y) 
print "--------------------------------------"


W=dot(pinv(train_x),train_y)
print "Transformation matrix shape : " , W.shape 

maxwidth=29
maxheight=32
deskewed_test=dot(test_x,W) 
err=(norm(test_y-deskewed_test)**2)/(float(maxwidth*maxheight*260))
print "L2 norm error for test set is : " , err

 
sample_dk=deskewed_test[150].reshape((maxwidth,maxheight))
sample_test=test_y[150].reshape((maxwidth,maxheight))
sample_test_sk=test_x[150].reshape((maxwidth,maxheight))

fig1=plt.figure()
subplot(131)
plt.xlabel('Skewed input')
plt.imshow(sample_test_sk,cmap='gray')
subplot(132)
plt.xlabel('Deskewed target') 
plt.imshow(sample_test,cmap='gray')
subplot(133)
plt.xlabel('Deskewed output') 
plt.imshow(sample_dk,cmap='gray')
plt.show()
fig1.savefig('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/results/linear_least_sq_'+str(angle)+'_new_white_on_black.jpg') 
plt.close("all")









