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



def convert_to_output(prd):
    prd=np.argmax(prd,axis=1)
    return prd


def confusion_matrix(prd,y):
    cm=np.zeros((classes,classes))
    for idx,i in enumerate(y):
        cm[i][prd[idx]]+=1
    return cm    



    
    
def compute_error(prd,y):
    # here this is mean square error
    y=convert_to_output(y)
    prd=convert_to_output(prd)
    return  np.power(((float(np.sum(np.multiply((prd-y),(prd-y))))/(2*y.shape[0]))),0.5)
    





def accuracy_rmse(prd,y):
    y=convert_to_output(y)
    prd=convert_to_output(prd)
    print "confusion matrix :: "
    print confusion_matrix(prd,y)
    
    return float(100*np.sum(prd==y))/y.shape[0]
    
    
def accuracy_average(prd,y):
    y=convert_to_output(y)
    prd=convert_to_output(prd)
    cls=np.zeros((classes,1))
    for i in range(classes):
        cls[i]=np.sum(y==i)
    acc=0
    for i in range(classes):
        if(cls[i] !=  0):
            acc= acc + ( np.sum(prd==i)/cls[i])
    return 100* np.divide(acc,(classes))


def accuracy_gm(prd,y):
    y=convert_to_output(y)
    prd=convert_to_output(prd)
    cls=np.zeros((classes,1))
    for i in range(classes):
        cls[i]=np.sum(y==i)
    
    acc=1;
    cnf=confusion_matrix(prd,y)
    for i in range(classes):
        if(cls[i] !=  0):
            a=np.sum(prd==i)
            acc= acc* (cnf[i,i]/float(cls[i]))

    return 100* np.power(acc,float(1/float(classes)))
    
def back_prop(xtrain,weights1,bias1,weights2,bias2,ytrain,alpha,yval,xval):
    for i in range(0,epochs):
        a1= np.dot(xtrain,np.transpose(weights1))+bias1
        h1=activation(a1)        
        a2= np.dot(h1,np.transpose(weights2))+bias2
        prd=activation(a2)
        zero_error=np.multiply(prd,ytrain)>1
        prd2=prd
        ytrain2=ytrain
        prd2[zero_error]=0
        ytrain2[zero_error]=0
        
        delta2=np.multiply(np.multiply((prd2-ytrain2),prd2),1- prd2 )
        dW2= np.dot(np.transpose(delta2),h1)  
        weights2=weights2 - alpha*dW2
        bias2=bias2-alpha*np.sum(delta2)
        delta1=(np.multiply(np.multiply(np.dot(delta2,weights2),h1),(1-h1)))        
        dW1= np.dot(np.transpose(delta1),xtrain)                
        weights1=weights1 - alpha*dW1
        bias1=bias1-alpha*np.sum(dW1)
        print compute_error(prd2,ytrain2)   
    ta1= np.dot(test,np.transpose(weights1))+bias1
    th1=activation(ta1)        
    ta2= np.dot(th1,np.transpose(weights2))+bias2
    tprd=activation(ta2)
        
    print "\n\nOverall accuracy ::\n"    
    print "on training data ::"
    print accuracy_rmse(prd2,ytrain2)
    print "\n"
    print "on test data ::"        
    print accuracy_rmse(tprd,answer)

    print "\n\nGeometric mean accuracy ::\n"    
    print "on training data ::"
    print accuracy_gm(prd2,ytrain2)
    print "\n"    
    print "on test data ::"        
    print accuracy_gm(tprd,answer)
    print "\n"


    print "\n\nAverage mean accuracy ::\n"    
    print "on training data ::"
    print accuracy_average(prd2,ytrain2)
    print "\n"    
    print "on test data ::"        
    print accuracy_average(tprd,answer)
    print "\n"


feat=5
classes=4
epochs=2000
alpha=0.001
hidden=100
train=open("Assignment Classification/Set 1/ae.tra")
test=open("Assignment Classification/Set 1/ae.tes")
answer=open("Assignment Classification/Results/Group 1/ae.cla")
answer=answer.read()

train=train.read()
test=test.read()
answer = [float(x) for x in answer.split()]


train = [float(x) for x in train.split()]
test = [float(x) for x in test.split()]

import numpy as np
train=np.array(train)
test=np.array(test)
answer=np.array(answer)


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
    


weights1=(np.random.rand(hidden,feat)-.5)/10
weights2=(np.random.rand(classes,hidden)-.5)/10
bias1=(np.random.rand(1,hidden)-.5)/10
bias2=(np.random.rand(1,classes)-.5)/10

answer=one_hot(answer)

ytrain=one_hot(ytrain)
yval=one_hot(yval)

# feature normalization
# the hero, savier in this code
xtrain=featureNormalize(xtrain)
xval=featureNormalize(xval)
test=featureNormalize(test)




    
back_prop(xtrain,weights1,bias1,weights2,bias2,ytrain,alpha,yval,xval)

        
    
    