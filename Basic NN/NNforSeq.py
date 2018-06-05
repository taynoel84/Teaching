import numpy as np


def sigmoid(X):
   return 1/(1+np.exp(-X))

netDescription=(2,20,10,1)


X=np.array([[1,2],[1,4],[3,1],[6,3],[3,6],[3,3],[3,4],[4,4]]) 

#Y=np.array([[1,    2,    3,    4,    5,    6,    7,    8]]).T #(2,5,5,1)
Y=np.array([[1,    2,    7,    5,    4,    6,    3,    8]]).T #(2,20,10,1)
#Y=np.array([[8,    2,    7,    5,    4,    6,    3,    1]]).T #(2,40,10,1)


niter=5000

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
   W.append(np.random.random((f,netDescription[n+1]))-1)
   B.append(np.zeros((1,netDescription[n+1])))
   x_delta.append([0])


TrainLoss=list()
for f in xrange(niter):
   #get input
   var1=int(np.floor(np.random.random()*len(X)))
   var2=-1
   while(var2<0 or var1==var2):
       var2=int(np.floor(np.random.random()*len(X)))
   seqvar1=Y[var1][0]
   seqvar2=Y[var2][0]
   if(seqvar1<seqvar2):
       x[0]=np.array([X[var1,:],X[var2,:]])
       y=np.array([[seqvar1,seqvar2]]).T
   else:
       x[0]=np.array([X[var2,:],X[var1,:]])
       y=np.array([[seqvar2,seqvar1]]).T
    
   #feed forward
   for f2 in xrange(numLayers-1):
      if(f2==numLayers-2):#output layer
          x[f2+1]=x[f2].dot(W[f2])+B[f2]
      else:
          x[f2+1]=sigmoid( x[f2].dot(W[f2])+B[f2])
          
   #Get output diff
   outDiff=x[numLayers-1][0]-x[numLayers-1][1]
   
   #get delta (dLoss/dLn) from backpropagation   
   for f2 in reversed(xrange(numLayers-1)):
      if f2==(numLayers-2):
         grad=1.0*(outDiff>0)
         x_delta[f2]=np.array([grad,-grad]) #dLoss/dL(top layer)
      else:
         x_delta[f2]=x_delta[f2+1].dot(W[f2+1].T)*(x[f2+1]*(1-x[f2+1]))
         
   #weight update      
   for f2 in xrange(numLayers-1):
      W[f2]-=x[f2].T.dot(x_delta[f2])
      B[f2]-=np.mean(x_delta[f2],axis=0).reshape(1,-1)
   
   



#Use
x[0]=X
for f2 in xrange(numLayers-1):
      if(f2==numLayers-2):#output layer
          x[f2+1]=x[f2].dot(W[f2])+B[f2]
      else:
          x[f2+1]=sigmoid( x[f2].dot(W[f2])+B[f2])

arr=np.concatenate((X,x[numLayers-1],Y),axis=1)
arr=arr[arr[:,2].argsort()]

print "Real_seq  Pred_seq  Input            Output"
for f in xrange(len(Y)):
    print  int(arr[f,3]),"       ",int(f+1),"       [",arr[f,0]," ",arr[f,1],"] ",arr[f,2]


x[0]=np.array([[2,1]])
if(True):
    for f2 in xrange(numLayers-1):
      if(f2==numLayers-2):#output layer
          x[f2+1]=x[f2].dot(W[f2])+B[f2]
      else:
          x[f2+1]=sigmoid( x[f2].dot(W[f2])+B[f2])
    