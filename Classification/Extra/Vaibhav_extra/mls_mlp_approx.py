# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:17:31 2016

@author: vaibh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 01:57:47 2016

@author: Vaibhav Amit Patel
"""

def featureNormalize(X):
    X_norm = np.divide((X-np.mean(X,axis=0)), np.std(X,axis=0))
    
    return X_norm

def activation(a):
    #here it is sigmoid function
    return np.divide(float(1),1+np.exp(-a))


def feed_forward(xtrain,weights1,bias1,weights2,bias2):
    a1= np.dot(xtrain,np.transpose(weights1))+bias1
    h1=activation(a1)        
    a2= np.dot(h1,np.transpose(weights2))+bias2
    prd=(a2)  
    return prd

    
    
def compute_error(prd,y):
    # here this is mean square error
    return  np.power(((float(np.sum(np.multiply((prd-y),(prd-y))))/(2*y.shape[0]))),0.5)
    
    
    
    
def back_prop(xtrain,weights1,bias1,weights2,bias2,ytrain,alpha,yval,xval):
    #prd=convert_to_output(feed_forward(xtrain,weights1,bias1,weights2,bias2))
    #print compute_error(prd,ytrain) 
    for i in range(0,epochs):
        a1= np.dot(xtrain,np.transpose(weights1))+bias1
        h1=activation(a1)        
        a2= np.dot(h1,np.transpose(weights2))+bias2
        prd=(a2)
        zero_error= np.multiply(prd,ytrain)>1 
        prd2=prd
        ytrain2=ytrain


        prd2[zero_error]=0
        ytrain2[zero_error]=0  
        delta2=(prd2-ytrain2)
        dW2= np.dot(np.transpose(delta2),h1)                
        weights2=weights2 - alpha*dW2
        bias2=bias2-alpha*np.sum(delta2)
        
        delta1=(np.multiply(np.multiply(np.dot(delta2,weights2),h1),(1-h1)))        
        dW1= np.dot(np.transpose(delta1),xtrain)                
        weights1=weights1 - alpha*dW1
        bias1=bias1-alpha*np.sum(dW1)
        print compute_error(prd2,ytrain2)   
    
    print "\n\nRMSE error ::\n"    
    print "on training data ::"
    print compute_error(prd2,ytrain2)   
    print "\n"
    print "on test data ::"        
    #print compute_error(xval,weights1,bias1,weights2,bias2,yval)   
    


feat=2
output=1
epochs=900
alpha=0.0001
hidden=600
train=open("her.tra")
test=open("her.tes")
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
    


weights1=(np.random.rand(hidden,feat)-.5)/1000
weights2=(np.random.rand(output,hidden)-.5)/1000
bias1=(np.random.rand(1,hidden)-.5)/1000
bias2=(np.random.rand(1,output)-.5)/1000



# feature normalization
# the hero, savier in this code
#xtrain=featureNormalize(xtrain)
#xval=featureNormalize(xval)


ytrain=np.reshape(ytrain,(ytrain.shape[0],1))    
back_prop(xtrain,weights1,bias1,weights2,bias2,ytrain,alpha,yval,xval)

        
    