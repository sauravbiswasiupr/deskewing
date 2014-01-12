'''MLP used from the ocrolib library of ocropus '''
from numpy import *
from pylab import *
from matplotlib import pyplot as plt
from PIL import Image
from numpy.linalg import pinv,norm
from ocrolib.lstm import *
import cPickle 
import argparse
__author__="Saurav"

parser=argparse.ArgumentParser()
parser.add_argument('hiddensize',type=int,help='number of hidden neurons to use')
parser.add_argument('skewangle',type=int,help='skew angle of images to be used')
parser.add_argument('learning_rate',type=float,help='learning rate to be used',default=1e-4)
parser.add_argument('numepochs',type=int,help='Number of epochs to use',default=1000)
parser.add_argument('--verbose',help='Verbosity turned on',action='store_true')
parser.add_argument('--display',help="Display images if asked otherwise not",action='store_true')
parser.add_argument('--binarize',help='Binarize the inputs and targets',action='store_true')
parser.add_argument('--randomization',help='shuffle around the dataset in each epoch',action='store_true')

args=parser.parse_args()
Nhidden=args.hiddensize
skew=args.skewangle
lr=args.learning_rate
epochs=args.numepochs
if args.verbose:
   print 'MLP that learns to deskew images'

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
        self.mlp=MLP(self.Ni,self.Nh,self.No)
        
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
        while tr_error >= 1e-4 and epoch < self.epochs:
           if args.randomization:
            tr_inputs,tr_targets=randomize(self.train_x,self.train_y)
            tst_inputs,tst_targets=randomize(self.test_x,self.test_y)
           else: 
            tr_inputs=self.train_x ; tr_targets=self.train_y
            tst_inputs=self.test_x  ; tst_targets=self.test_y 

           tr_outputs=array(self.mlp.forward(tr_inputs))
           tst_outputs=array(self.mlp.forward(tst_inputs))
           tst_deltas=tst_targets-tst_outputs
           tr_deltas=tr_targets-tr_outputs
           tr_error=norm(tr_deltas)**2/float(tr_deltas.shape[0]*tr_deltas.shape[1])
           tst_error=norm(tst_deltas)**2/float(tst_deltas.shape[0]*tst_deltas.shape[1])
           self.mlp.backward(tr_deltas)
           self.mlp.W1=self.mlp.W1+self.lr*self.mlp.DW1
           self.mlp.W2=self.mlp.W2+self.lr*self.mlp.DW2
            
            
            
           tr_error_list.append(tr_error)
           tst_error_list.append(tst_error)
           print "Epoch : " , (epoch+1) , " || Training  Error : " , tr_error , " || Test Error : " , tst_error
             
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
if args.binarize:
   print "Binarizing the data ..."
   train_x=1.0*(train_x != 0)
   train_y=1.0*(train_y != 0) 
   test_x=1.0*(test_x != 0)
   test_y=1.0*(test_y != 0)
maxwidth=29
maxheight=32
trainer=Trainer(maxwidth*maxheight,Nhidden,maxwidth*maxheight,lr) 


tr_error_list,tst_error_list=trainer.BackpropTrainer(train_x,train_y,test_x,test_y,epochs)
xs_train=[i for i in range(1,len(tr_error_list)+1)]
xs_test=[i for i in range(1,len(tst_error_list)+1)]
print "Testing on a random input target example"


inp=train_x[160]
targ=train_y[160] 

outp=array(trainer.mlp.forward([inp]))
print outp.shape 
outp=outp.T
outp=outp.reshape((maxwidth,maxheight))


'''final_outp=zeros((maxwidth,maxheight))
for i in range(maxwidth):
  for j in range(maxheight):
      if outp[i][j]>=0.5:
          final_outp[i][j]=outp[i][j]'''




fig=plt.figure()
subplot(131)
plt.xlabel('Skewed input')
plt.imshow(inp.reshape((maxwidth,maxheight)),cmap='gray')
subplot(132)
plt.xlabel('Deskewed Target')
plt.imshow(targ.reshape((maxwidth,maxheight)),cmap='gray')
subplot(133)
plt.xlabel('MLP Output') 
plt.imshow(outp,cmap='gray')
if args.display:
   plt.show()
fig.savefig('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/results/MLP_'+str(Nhidden)+'_hidden_skew_'+str(skew)+'_white_on_black.jpg')


fig1=plt.figure()
subplot(121)
plt.xlabel('Training Error')
plt.plot(xs_train,tr_error_list) 
subplot(122)
plt.xlabel('Test Error') 
plt.plot(xs_test,tst_error_list)
if args.display:
   plt.show()
fig1.savefig('/home/saurav/Desktop/CVPR_work/image_transformations/deskewing/results/plots_MLP_100_hidden_skew_'+str(skew)+'_white_on_black.jpg') 



 

