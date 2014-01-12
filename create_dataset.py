from numpy import *
from pylab import *
from PIL import Image 
from matplotlib import pyplot as plt
import os 
import cPickle 
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("skewangle",type=int,help="the angle by which we skew the images")
args=parser.parse_args()
angle=args.skewangle

def generate_random_matrix(width,height):
    '''Generates a random matrix for the given width and height '''
    random_matrix=random((width*height)).reshape((width,height))
    return random_matrix


def skewer(data,angle,width,height):
    '''Will skew an image by a distinct number of angles for each image in the dataset data'''
    images=[] 
    for d in data:
        im=d.reshape((width,height))
        im=Image.fromarray(im)
        rot=array(im.rotate(angle))
        rot=rot.reshape((width*height,))
        images.append(rot)
    images=array(images)
    return images

os.chdir('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/datasets/')
alphabets=[str(unichr(i)) for i in range(65,91)] 
assert len(alphabets) is 26
print alphabets 


images=[] ; widths=[] ; heights=[] 
for char in alphabets: 
      im=Image.open(str(char)+'_folder/fonts_standard_arial/010000.bin.png')
      im=asarray(im)
      images.append(im)
      w,h=im.shape
      widths.append(w) 
      heights.append(h)

maxwidth=max(widths) ;maxheight= max(heights)
padded=[]
for image in images:
    img=255*ones((maxwidth,maxheight))
    w,h=image.shape 
    widthToFill=maxwidth-w ; heightToFill=maxheight-h 
    wOffset=widthToFill/2 ; hOffset=heightToFill/2
    img[wOffset:wOffset+w , hOffset:hOffset+h]=image
    img=img.reshape((maxwidth*maxheight))
    padded.append(img)

padded=array(padded)
train_set=[] 
for i in range(len(padded)):
    img=padded[i].reshape((maxwidth,maxheight))
    for j in range(100):
        noise=generate_random_matrix(maxwidth,maxheight)
        
        noisy=img+noise
        noisy=noisy.reshape((maxwidth*maxheight))
        train_set.append(noisy)

train_set=array(train_set)/float(amax(train_set))
print amax(train_set) , amin(train_set)
print len(train_set)

assert len(train_set) == 2600
test_set=[]
new_train=[]
#now let's separate out 10 examples from each class to get our test set 
for j in range(1,27):
    chunk=train_set[-10+(100*j):100*j,:] 
    tr_chunk=train_set[100*(j-1):-10+(100*j)]
    for k in range(10):
        test_set.append(chunk[k])
    for l in range(90):
        new_train.append(tr_chunk[l])
print len(test_set)
print len(new_train)
test_set=array(test_set)/(float(amax(test_set)))
train_set=array(new_train)
train_set=train_set/(float(amax(train_set)))
train_set=1.0-train_set
test_set=1.0-test_set
print test_set[:,300:700]
#change to white on black form 
width,height=train_set.shape
train_x=zeros((width,height))
for i in range(width):
   for j in range(height):
      if train_set[i,j] >= 0.99:
         train_x[i,j] = train_set[i,j]
train_set=train_x
w,h=test_set.shape 
test_x=zeros((w,h))
for i in range(w):
   for j in range(h):
     if test_set[i,j]>=0.99:
        test_x[i,j]=test_set[i,j]

test_set=test_x
test_set=array(test_set)/(float(amax(test_set)))
train_set=train_set/(float(amax(train_set)))
print test_set.shape 
print "Test set max min :" , amax(test_set) , amin(test_set) 
print "Train set max min : ",amax(train_set) , amin(train_set)

shuffle(train_set)
shuffle(test_set)

print "Train_set length: " , len(train_set)


assert len(train_set)+len(test_set) == 2600
print "Skewing the training and test data by : " , angle , "angles"

train_sk=skewer(train_set,angle,maxwidth,maxheight)
test_sk=skewer(test_set,angle,maxwidth,maxheight)

train_skewed=zeros((width,height))
for i in range(width):
   for j in range(height):
      if train_sk[i,j] >= 0.99:
         train_skewed[i,j] = train_sk[i,j]
train_sk=train_skewed
w,h=test_set.shape 
test_skewed=zeros((w,h))
for i in range(w):
   for j in range(h):
     if test_sk[i,j]>=0.99:
        test_skewed[i,j]=test_sk[i,j]
test_sk=test_skewed

print "Train skewed length : " , len(train_sk)
'''train_set=train_set/float(amax(train_set)) ; train_sk=train_sk/float(amax(train_sk)) 
test_set=test_set/float(amax(test_set)) ; test_sk=test_sk/float(amax(test_sk))'''


f=open('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/datasets/train_'+str(angle)+'.pkl','wb')
train_sk=train_sk/(float(amax(train_sk)))
test_sk=test_sk/(float(amax(test_sk)))

print "Skewed train targets: " , amax(train_sk) , amin(train_sk)
print "Skewed test targets :" , amax(test_sk) , amin(test_sk)


train=train_sk,train_set
cPickle.dump(train,f)
f.close()
test
f=open('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/datasets/test_'+str(angle)+'.pkl','wb')
test=test_sk,test_set
cPickle.dump(test,f)
f.close()
