import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def sigmoid(X):
   return 1/(1+np.exp(-X))




def plot(X,y,Xt,yt):
    zerod=[]
    oned=[]
    for n,clas in enumerate(y):
        if clas[0]==0:
            zerod.append(X[n])
        else:
            oned.append(X[n])
    zerod=np.array(zerod)
    oned=np.array(oned)
    
    zerodt=[]
    onedt=[]
    for n,clas in enumerate(yt):
        if clas[0]==0:
            zerodt.append(Xt[n])
        else:
            onedt.append(Xt[n])
    zerodt=np.array(zerodt)
    onedt=np.array(onedt)
    plt.plot(zerod[:,0],zerod[:,1],'rx',oned[:,0],oned[:,1],'bx',
             zerodt[:,0],zerodt[:,1],'ro',onedt[:,0],onedt[:,1],'bo')

def plot2(X,y,):
    zerod=[]
    oned=[]
    for n,clas in enumerate(y):
        if clas[0]==0:
            zerod.append(X[n])
        else:
            oned.append(X[n])
    zerod=np.array(zerod)
    oned=np.array(oned)
    
    plt.ylim(ymax=10)
    plt.xlim(xmax=10)
    plt.plot(zerod[:,0],zerod[:,1],'ro',oned[:,0],oned[:,1],'bo')    
        
    
netDescription=(2,40,40,1)



#X=np.array([[0,0],[0,1],[1,0],[1,1]])
#y=np.array([[0,1,1,0]]).T

X=np.array([[4,5],[3.5,4.1],[6,6],[5,6],[1,2],[1,4],[3,1],[6,3],[3,6],[3,3],[3,4],[4,4],[4,6]])
y=np.array([[1,0,0,0,0,0,0,0,0,1,1,1,1]]).T

niter=10000

#build network structure
np.random.seed(100)
numLayers=len(netDescription)
W=list()
B=list()
x=list()
x_delta=list()
for n,f in enumerate(netDescription):
   x.append([0])
   if n==numLayers-1: break
   W.append(np.random.random((f,netDescription[n+1]))-0.5)
   B.append(np.zeros((1,netDescription[n+1])))
   x_delta.append([0])


x[0]=X

#a=bbb
TrainLoss=list()
for f in xrange(niter):
   #feed forward
   for f2 in xrange(numLayers-1):
      x[f2+1]=sigmoid( x[f2].dot(W[f2])+B[f2])
   
   #get delta (dLoss/dLn) from backpropagation   
   for f2 in reversed(xrange(numLayers-1)):
      if f2==(numLayers-2):
         x_delta[f2]=(x[f2+1]-y)*(x[f2+1]*(1-x[f2+1])) #dLoss/dL(top layer)
         x#_delta[f2]=-(y/x[f2+1]+((y-1)/(1-x[f2+1])))*(x[f2+1]*(1-x[f2+1]))
      else:
         x_delta[f2]=x_delta[f2+1].dot(W[f2+1].T)*(x[f2+1]*(1-x[f2+1]))
   
   #weight update      
   for f2 in xrange(numLayers-1):
      W[f2]-=x[f2].T.dot(x_delta[f2])/len(y)+0.0005*W[f2]
      B[f2]-=np.mean(x_delta[f2],axis=0).reshape(1,-1)
   
   #calc loss
   TrainLoss.append( -np.mean(np.log(x[numLayers-1])*y+(1-y)*np.log(1-x[numLayers-1])))

#Plot train loss
TrainLoss=np.array(TrainLoss)
#fig=plt.figure("1")
#plt.plot(TrainLoss)  

#Use
Xall=[]
yall=[]
for f in np.arange(1,10,0.2):
    for f2 in np.arange(1,10,0.2):
        x[0]=np.array([[f,f2]])
        for f3 in xrange(numLayers-1):
            x[f3+1]=sigmoid( x[f3].dot(W[f3])+B[f3])
        if(x[numLayers-1]<0.5):
            yall.append(np.array(0))
        else:
            yall.append(np.array(1))
        Xall.append(np.array([f,f2]))
Xall=np.array(Xall)
yall=np.array(yall).reshape(-1,1)
        
plot(Xall,yall,X,y)
