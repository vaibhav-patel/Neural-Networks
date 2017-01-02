# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 01:57:47 2016

@author: Vaibhav Amit Patel
"""

def featureNormalize(X):
    X_norm = np.divide((X-np.mean(X,axis=0)), np.std(X,axis=0))
    
    return X_norm

def rbf(a,mu,spread):
    #here it is RBF function
    a=np.exp(-np.sum(np.multiply((a-mu),(a-mu)),axis=1)/(2*spread*spread))
    return a
def activation(a):
    #here it is sigmoid function
    return np.divide(float(1),1+np.exp(-a))

def convert_to_output(prd):
    prd=np.argmax(prd,axis=1)
    return prd


def confusion_matrix(prd,y):
    cm=np.zeros((classes,classes))
    for idx,i in enumerate(y):
        cm[i][prd[idx]]+=1
    return cm    

  
    

def accuracy_rmse(prd,y):
    y=convert_to_output(y)
    prd=convert_to_output(prd)
    print "confusion matrix :: "
    print confusion_matrix(prd,y)
    
    return float(100*np.sum(prd==y))/y.shape[0]
    
def accuracy_gm(prd,y):
    y=convert_to_output(y)
    prd=convert_to_output(prd)
    cls=np.zeros((classes,1))
    for i in range(classes):
        cls[i]=np.sum(y==i)
    acc=1;
    for i in range(classes):
        if(cls[i] !=  0):
            acc= acc* ( np.sum(prd==i)/cls[i])
    return 100* np.power(acc,float(1/float(classes)))
    
def accuracy_average(prd,y):
    y=convert_to_output(y)
    prd=convert_to_output(prd)
    cls=np.zeros((classes,1))
    for i in range(classes):
        cls[i]=np.sum(y==i)
    acc=0;
    for i in range(classes):
        if(cls[i] !=  0):
            acc= acc+ ( np.sum(prd==i)/cls[i])
    return 100* np.divide(acc,(classes))
    
def back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K,spread):
    clusters= xtrain[np.random.choice(xtrain.shape[0], size=K,replace=False)]
    new_dim=np.zeros((xtrain.shape[0],K))
    if visualize:
        import  matplotlib.pyplot as plt
        plt.ion()
        plt.show()
        plt.plot(xtrain[:,0],xtrain[:,1],'o')
        plt.plot(clusters[:,0],clusters[:,1],'x')
        plt.draw()
    if visualize2:
	        import  matplotlib.pyplot as plt
	        plt.ion()
	        plt.show()
	        plt.draw()
    cnt=0
    for i in range(0,epochs):
        for i in range(K):
            new_dim[:,i]=  rbf(xtrain,clusters[i],spread)
        a1= np.dot(new_dim,np.transpose(weights1))+bias1
        prd=a1
        delta2=(prd-ytrain)
        dW2= np.dot(np.transpose(delta2), new_dim )
        delta_center=np.zeros((K,xtrain.shape[1]))
        for j in range(K):
            rp=np.repeat (new_dim[:,j],new_dim[:,j].shape[0], axis=0)
            rp=np.reshape(rp,(xtrain.shape[0],xtrain.shape[0]))
            delta_center[j] = np.dot (weights1.T[j,:], (np.dot( np.dot(delta2.T,  rp.T) , (xtrain-clusters[j])/(2*spread*spread))) )
            #print weights1.T[j,:].shape
            #print delta2.T.shape
            #print rp.shape
            #print clusters[j].shape
            #print delta_center.shape
            #raw_input()
        weights1=weights1 - alpha*dW2
        bias1=bias1-alpha*np.sum(delta2)
        clusters=clusters + 2*alpha_c*(delta_center)
        if visualize2:
        	plt.pause(0.05)
        	cnt=cnt+1
	        plt.plot(cnt,spread,'x')
	        plt.draw()
        #spread
        
        vap1=np.sum(new_dim,axis=1)
        """
        print vap1
        print vap1.shape
        raw_input()
        """
        arr=np.zeros((xtrain.shape[0],1))
        for i in range(K):
        	if spread != 0:
        		vap2=np.sum(np.power((xtrain-clusters[i]),2)/(spread*spread*spread), axis=1)
        		vap2=np.reshape(vap2,(vap2.shape[0],1))
        		arr=arr+vap2
    	vap1=np.reshape(vap1,(vap1.shape[0],1))
    	
    	arr=np.multiply(arr,vap1)
    	delta_spread= -2*np.dot(weights1.T, np.dot(delta2.T,arr))
    	delta_spread=np.sum(delta_spread)	
        spread=spread-alpha_spread*delta_spread
        print spread


        #spread

        #print delta_center
        #raw_input()
        if visualize:
            plt.pause(0.05)
            plt.plot(clusters[:,0],clusters[:,1],'x')
            plt.draw()
        print np.sum(np.multiply (prd-ytrain,prd-ytrain) )/ xtrain.shape[0]


    


visualize2=False
visualize=False
feat=5
import numpy
spread= 1
classes=4
epochs=1000
alpha=0.001
alpha_c=0.0001
alpha_spread=0.0000001

K=28
train=open("Assignment Classification/Set 1/ae.tra")
test=open("Assignment Classification/Set 1/ae.tes")
train=train.read()
test=test.read()


train = [float(x) for x in train.split()]
test = [float(x) for x in test.split()]

import numpy as np
train=np.array(train)
test=np.array(test)


def change_shape(x,f):    
    m=x.shape[0]
    return np.reshape(x,(m/f,f))

train=change_shape(train,feat+1)
test=change_shape(test,feat)


xtrain=train[:,0:feat]
ytrain=train[:,feat]

total_train=xtrain.shape[0]
total_val=int(total_train*0.1)

#10-fold validation
xval=xtrain[total_train-total_val:total_train,:]
xtrain=xtrain[0:total_train-total_val,:]
yval=ytrain[total_train-total_val:total_train]
ytrain=ytrain[0:total_train-total_val]
    
def one_hot(y):
    m=y.shape[0]
    narr=np.zeros((m,classes))
    y=y.astype(int)    
    for idx,i in enumerate(y):
        narr[idx][i-1]=1
    return narr
 

weights1=(np.random.rand(classes,K)-.5)/1000
bias1=(np.random.rand(1,classes)-.5)/1000

ytrain=one_hot(ytrain)
yval=one_hot(yval)

# feature normalization
# the hero, savier in this code
#xtrain=featureNormalize(xtrain)
#xval=featureNormalize(xval)

xtrain=featureNormalize(xtrain)
xval=featureNormalize(xval)



 
back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K,spread)

        
    
