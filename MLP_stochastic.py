'''MLP used from the ocrolib library of ocropus. Using the stochastic gradient descent approach in this method '''
from numpy import *
from pylab import *
from matplotlib import pyplot as plt
from PIL import Image
from numpy.linalg import pinv,norm
from ocrolib.lstm import *
import cPickle 
from random import choice 


__author__="Saurav"

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
        self.lr=lr 
        #self.train_data=train_data
        #self.target_data=target_data
        #self.epochs=epochs 
        self.mlp=MLP(self.Ni,self.Nh,self.No,initial_range=1)
    def BackpropTrainer(self,train_x,train_y,test_x,test_y,epochs):
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
        self.epochs=epochs
        
        print "Trainer initialized . Starting training ...... " 
        #using a batch approach initially. 
        #Stochastic Gradient descent implemented in this script
        tr_error_list=[]
        tst_error_list=[] 
        tst_error=1e10
        tr_error=1e10
        epoch=0
        indexes=[i for i in range(len(self.train_x))]
        while tr_error > 1e-4 and epoch < self.epochs:
            #tr_inputs,tr_targets=randomize(self.train_x,self.train_y)
            tst_inputs,tst_targets=randomize(self.test_x,self.test_y)
            index=choice(indexes)
            input=self.train_x[index] 
            target=self.train_y[index]
            tr_output=array(self.mlp.forward([input]))
            tst_outputs=array(self.mlp.forward(tst_inputs))
            tst_deltas=tst_targets-tst_outputs
            tr_delta=target-tr_output
            print len(tr_delta)
            #print tr_delta.shape
            tr_error=norm(tr_delta)**2/float(tr_delta.shape[0]*tr_delta.shape[1])
            tst_error=norm(tst_deltas)**2/float(tst_deltas.shape[0]*tst_deltas.shape[1])
            tr_delta=[tr_delta]
            print len(tr_delta)
            self.mlp.backward(tr_delta)
            self.mlp.W1=self.mlp.W1+self.lr*self.mlp.DW1
            self.mlp.W2=self.mlp.W2+self.lr*self.mlp.DW2
            
            
            
            tr_error_list.append(tr_error)
            tst_error_list.append(tst_error)
            print "Epoch : " , (epoch+1) , " || Training  Error : " , tr_error , " || Test Error : " , tst_error
             
            epoch=epoch+1
    
        return tr_error_list,tst_error_list

print "Loading the dataset ..." 

f=open('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/datasets/train_5.pkl','rb')
train_x,train_y=cPickle.load(f)
assert train_x.shape[0] == 2340

f.close()
f=open('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/datasets/test_5.pkl','rb')
test_x,test_y=cPickle.load(f)
assert test_x.shape[0] == 260
f.close()

print "Dataset loaded successfully..."  

maxwidth=29
maxheight=32
trainer=Trainer(maxwidth*maxheight,50,maxwidth*maxheight,1e-4) 
epochs=1500

train_x=train_x/amax(train_x) ; train_y=train_y/amax(train_y)
test_x=test_x/amax(test_x) ; test_y=test_y/amax(test_y)

tr_error_list,tst_error_list=trainer.BackpropTrainer(train_x,train_y,test_x,test_y,epochs)
xs_train=[i for i in range(1,len(tr_error_list)+1)]
xs_test=[i for i in range(1,len(tst_error_list)+1)]
print "Testing on a random input target example"


inp=test_x[160]
targ=test_y[160] 

outp=array(trainer.mlp.forward([inp]))
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
fig.savefig('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/results/MLP_50_hidden_skew_5_stochastic.jpg')


fig1=plt.figure()
subplot(121)
plt.xlabel('Training Error')
plt.plot(xs_train,tr_error_list) 
subplot(122)
plt.xlabel('Test Error') 
plt.plot(xs_test,tst_error_list)
fig1.savefig('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/results/plots_MLP_50_hidden_skew_5_stochastic.jpg') 
show()



 

