#!/usr/bin/python 
##Network that takes as input, a sequence of decreasingly skewed images(unrolled 1D) and 
##learns to output deskewed versions 

#fix imports 
from numpy import *
from pylab import *
from matplotlib import pyplot as plt
import cPickle 
from numpy.linalg import norm 
from PIL import Image 
import argparse
from ocrolib.lstm import *
import os

__author__="Saurav"

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


parser=argparse.ArgumentParser()
#add some parser arguments
parser.add_argument('hiddenSize',type=int,help='Number of hidden neurons in LSTM layer')
parser.add_argument('epochs',type=int,help='Number of epochs')
parser.add_argument('skews',type=int,nargs='+',help='Skews to be used') 
parser.add_argument('--display',help='Display the plots if wanted',action='store_true')

args=parser.parse_args()
nhidden=args.hiddenSize
epochs=args.epochs
skews=args.skews


f=open('../datasets/train_0.pkl','rb')
train_0_x,train_0_y=cPickle.load(f)
f.close()

f=open('../datasets/test_0.pkl','rb')
test_0_x,test_0_y=cPickle.load(f)
f.close()


print "Unskewed data loaded ..." , train_0_x.shape 
print "Creating input sequences with a variety of skews ..."

def inputSequences(inps,targs,skews):
    #skews=[-3,-2,-1,0] 
    train_x=[]
    train_y=[]
    for skew in skews:
      t_x=skewer(inps,skew,29,32)
      t_y=skewer(targs,0,29,32)   #targets are unskewed for all timesteps == len(skews)
      train_x.append(t_x)
      train_y.append(t_y)
    w,h=inps.shape 
    train_set_x=[]
    train_set_y=[]
    for i in range(w):
         '''c=hstack((train_x[0][i].reshape((928,1)),train_x[1][i].reshape((928,1)),train_x[2][i].reshape((928,1)),train_x[3][i].reshape((928,1))))
         train_set_x.append(c.T)     
         d=hstack((train_y[0][i].reshape((928,1)),train_y[1][i].reshape((928,1)),train_y[2][i].reshape((928,1)),train_y[3][i].reshape((928,1))))
         train_set_y.append(d.T)'''
         c=train_x[0][i].reshape((928,1))
         for j in range(1,len(skews)):
            c=hstack((c,train_x[j][i].reshape((928,1))))
         train_set_x.append(c.T)
         d=train_y[0][i].reshape((928,1))
         for j in range(1,len(skews)):
            d=hstack((d,train_y[j][i].reshape((928,1))))
         train_set_y.append(d.T)
    train_set_x=array(train_set_x)
    train_set_y=array(train_set_y)
    return train_set_x,train_set_y 

class Trainer:
    def __init__(self,Ni,Nh,No,lr):
        self.Ni=Ni 
        self.Nh=Nh
        self.No=No 
        #self.train_data=train_data
        #self.target_data=target_data
        #self.epochs=epochs 
        self.lstm=LSTM(self.Ni,self.Nh)
        self.logreg=Logreg(self.Nh,self.No)
        self.net=Stacked([self.lstm,self.logreg])
        self.net.momentum=0.9
        self.net.learning_rate=lr
    def BackpropTrainer(self,train_x,train_y,epochs):
        self.train_x=train_x
        self.train_y=train_y
        self.epochs=epochs
        
        print "Trainer initialized . Starting training ...... " 
        #using a batch approach initially. 
        #SGD TODO
        tr_error_list=[]
        
        tr_error=1e10
        epoch=0
        while tr_error >= 1e-4 and epoch < self.epochs:
            tr_inputs,tr_targets=train_x,train_y
            #tst_inputs,tst_targets=randomize(self.test_x,self.test_y)
            train_error=0 ; test_error=0   #initialize the training and test error to 0 before each epoch starts
            for sample,targ in zip(tr_inputs,tr_targets):
               tr_output=array(self.net.forward(sample))
               #tst_outputs=array(self.mlp.forward(tst_inputs))
               #tst_deltas=tst_outputs-tst_targets
               tr_delta=targ-tr_output
               tr_error=norm(tr_delta)**2/float(tr_delta.shape[0]*tr_delta.shape[1])
               self.net.backward(tr_delta)
               self.net.update()
               train_error=train_error+tr_error
            '''for t_sample,t_targ in zip(tst_inputs,tst_targets):
               tst_output=array(self.net.forward([t_sample]))
               tst_delta=t_targ-tst_output
               tst_error=norm(tst_delta)**2/float(tst_delta.shape[0]*tst_delta.shape[1])
               test_error=test_error+tst_error'''
           
            
            train_error=train_error/float(len(tr_inputs))
            #test_error=test_error/float(len(tst_inputs))
            tr_error_list.append(train_error)
            #tst_error_list.append(test_error)
            print "Epoch : " , (epoch+1) , " || Training  Error : " , train_error #, " || Test Error : " , test_error
             
            epoch=epoch+1
    
        return tr_error_list #,tst_error_list

print "Skews being used : " , skews 
train_set_x,train_set_y=inputSequences(train_0_x,train_0_y,skews)
test_set_x,test_set_y=inputSequences(test_0_x,test_0_y,skews)

print "Inputsequences created for the LSTM..."

trainer=Trainer(928,nhidden,928,1e-4)
tr_error=trainer.BackpropTrainer(train_set_x,train_set_y,epochs)
print "Showing some sample outputs..."

target=test_set_y[30]
inp=test_set_x[30]
predicted=array(trainer.net.forward(inp))[-1]  #take the last timestep value 


fig=plt.figure()
for i in range(len(skews)):
     command=str(1)+str(len(skews)+1)+str(i+1)
     #print "subplot parameters : " , command 
     plt.subplot(int(command))
     plt.imshow(inp[i].reshape((29,32)),cmap='gray')
     plt.axis('off')
cmd=str(1)+str(len(skews)+1)+str(len(skews)+1)
plt.subplot(int(cmd))
plt.axis('off')
plt.imshow(predicted.reshape((29,32)),cmap='gray')
fig.savefig('../results/deskewing_LSTM_sequences/LSTM_deskewed_'+str(nhidden)+'_hidden_'+str(len(skews))+'_skewlistlength_'+str(max(skews))+'_skew.jpg')
if args.display:
    plt.show()



