#!/usr/bin/python 
##Script that tries to learn the amount of skew in an image and uses a softmax to make a 1 out of n skew classification 

#fix imports 
from numpy import *
from pylab import *
from matplotlib import pyplot as plt 
from numpy.linalg import norm
import cPickle 
from ocrolib.lstm import *
from PIL import Image 
from numpy import log 
import argparse
__author__="Saurav"

parser=argparse.ArgumentParser()
parser.add_argument('hiddensize',type=int,help='number of hidden neurons in the hidden layer')
parser.add_argument('epochs',type=int,help='number of epochs') 
parser.add_argument('skews',type=int,nargs='+',help='list of skews to use')

args=parser.parse_args()
nhidden=args.hiddensize 
epochs=args.epochs
skews=args.skews

class Trainer:
  def __init__(self,Ni,Nh,No,epochs,lr):
      self.Ni=Ni
      self.No=No
      self.epochs=epochs
      self.lr=lr
      self.Nh=Nh
      self.mlp=Logreg(self.Ni,self.Nh)
      self.softmax=Softmax(self.Nh,self.No)
      self.net=Stacked([self.mlp,self.softmax])
      self.net.setLearningRate(self.lr,0.9)
  def BatchTrainer(self,train_x,train_y,train_targets,maxskew):
      
      self.train_x=train_x
      self.train_y=train_y
      self.train_targets=train_targets
      self.maxskew=maxskew
      #fwd propagate the inputs 
      epoch=0
      tr_errors=[]
      #tst_errors=[]
      tr_error=1e2
      print "BatchTrainer initialized ..."
      print "Starting Training ..."
      while epoch < self.epochs and tr_error >=1e-4 :
       
            
         #train_x,train_y=randomize(self.train_x,self.train_y)
    
         train_x,train_y=self.train_x,self.train_y
         outp=array(self.net.forward(train_x))
         targs=train_y
         deltas=targs-outp
         #print tr_deltas.shape , len(tr_inps)
         '''error=-sum(targs*log(outp))
         error=error/float(len(outp))'''
         predictions=argmax(outp,axis=1)-maxskew
         error=1*(self.train_targets != predictions)
         error=sum(error)/float(len(error))
         tr_errors.append(error)
         print "Epoch : ", epoch , "|| Training Error : " , error
         #tst_errors.append(tst_error)
         self.net.backward(deltas)
         self.net.update()
         epoch=epoch+1
      return tr_errors

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

def createInputsTargets(x,skews,width,height):
   '''Function that creates a dataset of shuffled (img,skew) pairs from an initially unskewed dataset'''
   train_x=[]
   train_y=[]
   labels=[]
   for skew in skews:
      skewed_x=skewer(x,skew,width,height)
      for im in skewed_x:
         train_x.append(im)
         #the skew is the label so it will be mapped such that 
         label=zeros((len(skews),))
         index=skew+max(skews)
         label[index]=1
         train_y.append(label)
         labels.append(skew)

   assert len(train_x) == len(skews)*len(x)
   train_x=array(train_x)
   train_y=array(train_y)
   labels=array(labels)
   indices=[i for i in range(len(train_x))]
   shuffle(indices)
   inp_x=[]
   inp_y=[]
   targets=[]
   for i in indices:
     inp_x.append(train_x[i])
     inp_y.append(train_y[i])
     targets.append(labels[i])

   inp_x=array(inp_x)
   inp_y=array(inp_y)
   targets=array(targets)
   return inp_x,inp_y,targets


#open the base images (unskewed) 
f=open('../datasets/train_0.pkl','rb')
train_0_x,train_0_y=cPickle.load(f)
f.close()

f=open('../datasets/test_0.pkl','rb')
test_0_x,test_0_y=cPickle.load(f)
f.close()

print "Unskewed data loaded ..."
print "Skews being used : " , skews 
print "Creating the Inputs and Targets ..."

train_x,train_y,train_targets=createInputsTargets(train_0_x,skews,29,32)
test_x,test_y,test_targets=createInputsTargets(test_0_x,skews,29,32)

print "Initializing the trainer..."
trainer=Trainer(928,nhidden,len(skews),epochs,1e-4)
tr_errors=trainer.BatchTrainer(train_x,train_y,train_targets,max(skews))

test_pred=array(trainer.net.forward(test_x))
predictions=argmax(test_pred,axis=1)-max(skews)
loss=1*(test_targets != predictions)
test_error= sum(loss)/float(len(loss))
print "Test Set error : " , test_error
f=open('../results/MLP_hidden_'+str(nhidden)+'_maxskew_'+str(max(skews))+'_softmax.txt','w')
f.write('Test Set error : '+str(test_error))
