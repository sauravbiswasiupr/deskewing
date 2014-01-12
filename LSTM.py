'''LSTM used from the ocrolib library of ocropus '''
from numpy import *
from pylab import *
from matplotlib import pyplot as plt
from PIL import Image
from numpy.linalg import pinv,norm
from ocrolib.lstm import *
import cPickle 
import argparse


__author__="Saurav"

#add some parsing arguments 
parser=argparse.ArgumentParser() ;
parser.add_argument('hiddensize',type=int,help='number of hidden LSTM nodes to use') 
parser.add_argument('skewangle',type=int,help='the skewing that we need to learn')
parser.add_argument('learning_rate',type=float,help='The learning rate for the LSTM network',default=1e-4)
parser.add_argument('numepochs',type=int,help='The number of epochs that the network would run',default=1000) 
parser.add_argument('--display',help='Display images if asked otherwise not',action='store_true') 
parser.add_argument('--verbose',help='Verbosity turned on',action='store_true')

args=parser.parse_args()
Nhidden=args.hiddensize
skew=args.skewangle
lr=args.learning_rate
epochs=args.numepochs


def randomize(train_x,train_y):
    #randomize the train_x and train_y examples each epoch
    assert len(train_x) == len(train_y)
    indexes=[i for i in range(len(train_x))]
    shuffle(indexes) 
    width=train_x.shape[0] 
    height=train_x.shape[1]
    copy_x=zeros((width,height))
    copy_y=zeros((width,height))
    for ind in indexes:
      copy_x[i]=train_x[i] 
      copy_y[i]=train_y[i] 
    return copy_x,copy_y

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
    def BackpropTrainer(self,train_x,train_y,test_x,test_y,epochs):
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
        self.epochs=epochs
        
        print "Trainer initialized . Starting training ...... " 
        #using a batch approach initially. 
        #SGD TODO
        tr_error_list=[]
        tst_error_list=[] 
        tst_error=1e10
        tr_error=1e10
        epoch=0
        while tst_error >= 1e-7 and epoch < self.epochs:
            tr_inputs,tr_targets=randomize(self.train_x,self.train_y)
            tst_inputs,tst_targets=randomize(self.test_x,self.test_y)
            train_error=0 ; test_error=0   #initialize the training and test error to 0 before each epoch starts
            for sample,targ in zip(tr_inputs,tr_targets):
               tr_output=array(self.net.forward([sample]))
               #tst_outputs=array(self.mlp.forward(tst_inputs))
               #tst_deltas=tst_outputs-tst_targets
               tr_delta=targ-tr_output
               tr_error=norm(tr_delta)**2/float(tr_delta.shape[0]*tr_delta.shape[1])
               self.net.backward(tr_delta)
               self.net.update()
               train_error=train_error+tr_error
            for t_sample,t_targ in zip(tst_inputs,tst_targets):
               tst_output=array(self.net.forward([t_sample]))
               tst_delta=t_targ-tst_output
               tst_error=norm(tst_delta)**2/float(tst_delta.shape[0]*tst_delta.shape[1])
               test_error=test_error+tst_error
           
            
            train_error=train_error/float(len(tr_inputs))
            test_error=test_error/float(len(tst_inputs))
            tr_error_list.append(train_error)
            tst_error_list.append(test_error)
            print "Epoch : " , (epoch+1) , " || Training  Error : " , train_error , " || Test Error : " , test_error
             
            epoch=epoch+1
    
        return tr_error_list,tst_error_list

print "Loading the dataset ..." 

f=open('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/datasets/train_'+str(skew)+'.pkl','rb')
train_x,train_y=cPickle.load(f)
assert train_x.shape[0] == 2340

f.close()
f=open('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/datasets/test_'+str(skew)+'.pkl','rb')
test_x,test_y=cPickle.load(f)
assert test_x.shape[0] == 260
f.close()

print "Dataset loaded successfully..."  

maxwidth=29
maxheight=32
trainer=Trainer(maxwidth*maxheight,Nhidden,maxwidth*maxheight,lr) 


train_x=train_x/amax(train_x) ; train_y=train_y/amax(train_y)
test_x=test_x/amax(test_x) ; test_y=test_y/amax(test_y)

tr_error_list,tst_error_list=trainer.BackpropTrainer(train_x,train_y,test_x,test_y,epochs)
xs_train=[i for i in range(1,len(tr_error_list)+1)]
xs_test=[i for i in range(1,len(tst_error_list)+1)]
print "Testing on a random input target example"


inp=test_x[160]
targ=test_y[160] 

outp=array(trainer.net.forward([inp]))
print outp.shape 
outp=outp.T


fig=plt.figure()
subplot(131)
plt.xlabel('Skewed input')
plt.imshow(inp.reshape((maxwidth,maxheight)),cmap='gray')
subplot(132)
plt.xlabel('Deskewed Target')
plt.imshow(targ.reshape((maxwidth,maxheight)),cmap='gray')
subplot(133)
plt.xlabel('MLP Output') 
plt.imshow(outp.reshape((maxwidth,maxheight)),cmap='gray')
fig.savefig('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/results/LSTM_'+str(Nhidden)+'_hidden_skew_'+str(skew)+'.jpg')


fig1=plt.figure()
subplot(121)
plt.xlabel('Training Error')
plt.plot(xs_train,tr_error_list) 
subplot(122)
plt.xlabel('Test Error') 
plt.plot(xs_test,tst_error_list)
fig1.savefig('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/results/plots_LSTM_'+str(Nhidden)+'_hidden_skew_'+str(skew)+'.jpg') 
if args.display:
   plt.show()



 

