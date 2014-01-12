from numpy import *
from pylab import *
from matplotlib import pyplot as plt
from ocrolib.lstm import *
from PIL import Image
import cPickle 
import argparse

parser=argparse.ArgumentParser()
#add some parser arguments 
parser.add_argument('skewangle',type=int,help='Skewed images that need to be used are defined by train_<skew_angle>.pkl')
parser.add_argument('hiddensize1',type=int,help='Number of hidden neurons in first hidden layer')
parser.add_argument('hiddensize2',type=int,help='Number of hidden neurons in second hidden layer')
parser.add_argument('learning_rate',type=float,help='Learning rate to be used for the network',default=1e-4)
parser.add_argument('numepochs',type=int,help='The max number of epochs that you want your network to run for',default=1000)
parser.add_argument('--display',help='Display the plots if you want to',action='store_true')
parser.add_argument('--randomization',help='randomize the data values if you want during each epoch',action='store_true')

args=parser.parse_args()
Nh1=args.hiddensize1
Nh2=args.hiddensize2
lr=args.learning_rate
epochs=args.numepochs
skew=args.skewangle

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
  def __init__(self,Ni,Nh1,Nh2,No,epochs,lr):
      self.Ni=Ni
      self.No=No
      self.epochs=epochs
      self.lr=lr
      self.Nh1=Nh1
      self.Nh2=Nh2
      self.mlp=MLP(self.Ni,self.Nh1,self.Nh2)
      self.logreg=Logreg(self.Nh2,self.No)
      self.net=Stacked([self.mlp,self.logreg])
      self.net.setLearningRate(self.lr,0.9)
  def BatchTrainer(self,train_x,train_y):
      if args.randomization:
         self.train_x,self.train_y=randomize(train_x,train_y)
      else:
         self.train_x=train_x
         self.train_y=train_y
      #fwd propagate the inputs 
      epoch=0
      tr_errors=[]
      #tst_errors=[]
      tr_error=1e2
      print "BatchTrainer initialized ..."
      print "Starting Training ..."
      while epoch < self.epochs and tr_error >=1e-5 :
         outp=array(self.net.forward(self.train_x))
         targs=self.train_y
         deltas=targs-outp
         #print tr_deltas.shape , len(tr_inps)
         error=norm(deltas)**2/float(deltas.shape[0]*deltas.shape[1])
         print "Epoch : ", epoch , "|| Training Error : " , error 
         tr_errors.append(error)
         #tst_errors.append(tst_error)
         self.net.backward(deltas)
         self.net.update()
         epoch=epoch+1
      return tr_errors

print "Loading the datasets ..."
f=open('../datasets/train_'+str(skew)+'.pkl','rb')
train_x,train_y=cPickle.load(f)
f.close()

f=open('../datasets/test_'+str(skew)+'.pkl','rb')
test_x,test_y=cPickle.load(f)
f.close()

print "Dataset loaded ..."
maxwidth=29 ;maxheight=32 
trainer=Trainer(maxwidth*maxheight,Nh1,Nh2,maxwidth*maxheight,epochs,lr)
training_errors=trainer.BatchTrainer(train_x,train_y)



inp=test_x[120]
targ=test_y[120]
outp=array(trainer.net.forward([inp]))
outp=outp.T

inp=inp.reshape((maxwidth,maxheight))
outp=outp.reshape((maxwidth,maxheight))
targ=targ.reshape((maxwidth,maxheight))

fig=plt.figure()
plt.subplot(131)
plt.xlabel('Skewed input')
plt.imshow(inp,cmap='gray')
plt.subplot(132)
plt.xlabel('Deskewed Target')
plt.imshow(targ,cmap='gray')
plt.subplot(133)
plt.xlabel('MLP output')
plt.imshow(outp,cmap='gray')
if args.display:
   plt.show()
fig.savefig('../results/MLP_multiHiddenLayer_L1_'+str(Nh1)+'_L2_'+str(Nh2)+'_'+str(skew)+'_angles.jpg')


        
         
