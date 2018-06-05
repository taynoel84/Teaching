import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(100)

def sigmoid(X):
   return 1/(1+np.exp(-X))

netDescription=(1,35,1)

##polynomial example
px=np.array(xrange(50))-25
py=-0.00004*(px**4)-0.0015*(px**3)+0.02*(px**2)+0.5*px+2

X=np.array([[-24],[-3.],[-5],[24]])
y=np.array([[10,1.75,-2,-7.5]]).T

X=np.array([
         [px[10]],[px[15]],[px[25]],[px[3]],
         [px[35]],[px[20]],[px[30]],[px[40]],[px[45]]
         #[px[2]],[px[7]],[px[12]],[px[17]],[px[22]],
         #[px[3]],[px[8]],[px[13]],[px[18]],[px[23]],
         #[px[28]],[px[33]],[px[38]],[px[43]],[px[48]],
         #[px[31]],[px[36]],[px[41]],[px[34]],[px[42]],
         #[px[24]],
         #[px[2]],[px[7]],[px[12]],[px[17]],[px[22]],
         #[px[3]],[px[8]],[px[13]],[px[18]],[px[23]]
        ])
y=np.array([[
             py[10]+5*(np.random.random(1)[0]-0.5),
             py[15]+5*(np.random.random(1)[0]-0.5),
             py[25]+5*(np.random.random(1)[0]-0.5),
             py[3]+5*(np.random.random(1)[0]-0.5),
             py[35]+5*(np.random.random(1)[0]-0.5),
             py[20]+5*(np.random.random(1)[0]-0.5),
             py[30]+5*(np.random.random(1)[0]-0.5),
             py[40]+5*(np.random.random(1)[0]-0.5),
             py[45]+5*(np.random.random(1)[0]-0.5)
             #py[2]+5*(np.random.random(1)[0]-0.5),
             #py[7]+5*(np.random.random(1)[0]-0.5),
             #py[12]+5*(np.random.random(1)[0]-0.5),
             #py[17]+5*(np.random.random(1)[0]-0.5),
             #py[22]+5*(np.random.random(1)[0]-0.5),
             #py[3]+5*(np.random.random(1)[0]-0.5),
             #py[8]+5*(np.random.random(1)[0]-0.5),
             #py[13]+2*(np.random.random(1)[0]-0.5),
             #py[18]+2*(np.random.random(1)[0]-0.5),
             #py[23]+2*(np.random.random(1)[0]-0.5),
             #py[28]+2*(np.random.random(1)[0]-0.5),
             #py[33]+2*(np.random.random(1)[0]-0.5),
             #py[38]+2*(np.random.random(1)[0]-0.5),
             #py[43]+2*(np.random.random(1)[0]-0.5),
             #py[48]+2*(np.random.random(1)[0]-0.5),
             #py[31]+2*(np.random.random(1)[0]-0.5),
             #py[36]+2*(np.random.random(1)[0]-0.5),
             #py[41]+2*(np.random.random(1)[0]-0.5),
             #py[34]+2*(np.random.random(1)[0]-0.5),
             #py[42]+2*(np.random.random(1)[0]-0.5),
             #py[24]+2*(np.random.random(1)[0]-0.5),
             #py[2]+5*(np.random.random(1)[0]-0.5),
             #py[7]+5*(np.random.random(1)[0]-0.5),
             #py[12]+5*(np.random.random(1)[0]-0.5),
             #py[17]+5*(np.random.random(1)[0]-0.5),
             #py[22]+5*(np.random.random(1)[0]-0.5),
             #py[3]+5*(np.random.random(1)[0]-0.5),
             #py[8]+5*(np.random.random(1)[0]-0.5),
             #py[13]+2*(np.random.random(1)[0]-0.5),
             #py[18]+2*(np.random.random(1)[0]-0.5),
             #py[23]+2*(np.random.random(1)[0]-0.5)
             ]]).T

niter=50000

#build network structure

numLayers=len(netDescription)
W=list()
B=list()
x=list()
x_delta=list()
for n,f in enumerate(netDescription):
   x.append([0])
   if n==numLayers-1: break
   W.append(np.random.random((f,netDescription[n+1]))*5)
   B.append(np.zeros((1,netDescription[n+1])))
   x_delta.append([0])


x[0]=X


for f in xrange(niter):
   #feed forward
   for f2 in xrange(numLayers-1):
      if f2==(numLayers-2):
         x[f2+1]= x[f2].dot(W[f2])+B[f2]
      else:
         x[f2+1]=sigmoid( x[f2].dot(W[f2])+B[f2])
   
   #get delta (dLoss/dLn) from backpropagation   
   for f2 in reversed(xrange(numLayers-1)):
      if f2==(numLayers-2):
         x_delta[f2]=(x[f2+1]-y) #dLoss/dL(top layer)
         #x_delta[f2]=-(y/x[f2+1]+((y-1)/(1-x[f2+1])))*(x[f2+1]*(1-x[f2+1]))
      else:
         x_delta[f2]=x_delta[f2+1].dot(W[f2+1].T)*(x[f2+1]*(1-x[f2+1]))
   
   #weight update      
   for f2 in xrange(numLayers-1):
      W[f2]-=0.02*(x[f2].T.dot(x_delta[f2])/len(y)+0.1*W[f2])
      B[f2]-=0.02*np.mean(x_delta[f2],axis=0).reshape(1,-1)
   



#Use
x[0]=X
for f2 in xrange(numLayers-1):
      if f2==(numLayers-2):
         x[f2+1]= x[f2].dot(W[f2])+B[f2]
      else:
         x[f2+1]=sigmoid( x[f2].dot(W[f2])+B[f2])
         
def func(Xin):
    x[0]=np.array([[Xin]])
    for f2 in xrange(numLayers-1):
      if f2==(numLayers-2):
         x[f2+1]= x[f2].dot(W[f2])+B[f2]
      else:
         x[f2+1]=sigmoid( x[f2].dot(W[f2])+B[f2])
    return x[numLayers-1]

#for drawing
xx=[]
for f in xrange(50):
    xx.append(func(f-25))
xx=np.array(xx)
#plt.plot(np.array(xrange(50))-25,xx[:,0,0])


plt.ylim(ymax=15,ymin=-10)
plt.plot(X,y,'ro',
         np.array(xrange(50))-25,xx[:,0,0],'g--',
         np.array(xrange(50))-25,py,'b')