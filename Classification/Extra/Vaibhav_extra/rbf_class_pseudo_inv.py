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
    

def k_means(xtrain,K):
    clusters= xtrain[np.random.choice(xtrain.shape[0], size=K,replace=False)]
    err=999;
    if visualize:
        import  matplotlib.pyplot as plt
        plt.ion()
        plt.show()
        plt.plot(xtrain[:,0],xtrain[:,1],'o')
        plt.plot(clusters[:,0],clusters[:,1],'x')
        plt.draw()

    olderr=err+100
    while err>=0.5: 
        dis=np.zeros((K,xtrain.shape[0]))
        for i in range(K):
            dis[i]= np.sum(np.multiply((xtrain - clusters[i]),((xtrain - clusters[i]))),axis=1)
        assignment = np.argmin(dis,axis=0)

        oldones=clusters
        for i in range(K):
            tot_in_c=float (np.sum(assignment==i))
            if tot_in_c>0:
                clusters[i]= np.sum(xtrain[assignment==i],axis=0)/tot_in_c
        err=0
        for i in range(K):
            err= err + np.sum(np.multiply((xtrain[assignment==i] - clusters[i]),((xtrain[assignment==i] - clusters[i]))))
        if visualize:               
            print err
            plt.pause(0.5)
            for i in range(K):
                vap=xtrain[assignment==i]
                plt.plot(vap[:,0],vap[:,1],'o')
            plt.plot(clusters[:,0],clusters[:,1],'x')
            plt.draw()
        if(np.absolute(olderr-err) <= 0.01):
            break;
        olderr=err

    return clusters
    
def back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K):
	clusters=k_means(xtrain,K)
    #clusters= xtrain[np.random.choice(xtrain.shape[0], size=K,replace=False)]
	cls_dis=np.zeros((K,K))
	for i in range(K):
		for j in range(K):
			cls_dis[i,j]=  np.power(np.sum(np.multiply(clusters[i]-clusters[j],clusters[i]-clusters[j])),0.5)
	max_d= np.max(cls_dis)
	spread=max_d/float (np.power(K,0.5))
	new_dim=np.zeros((xtrain.shape[0],K))
	for i in range(K):
		new_dim[:,i]=  rbf(xtrain,clusters[i],spread)
	saved_dim=new_dim
	new_dim=np.concatenate((new_dim,np.ones((new_dim.shape[0],1))),axis=1)
	answer= np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(new_dim),new_dim)),np.transpose(new_dim)),ytrain)
	print answer
	weights1=answer[0:answer.shape[0]-1, :]
	bias1=answer[-1,:]
	weights1=weights1.T
	bias1=bias1.T
	a1= np.dot(saved_dim,np.transpose(weights1))+bias1
	prd=(a1)
	print np.sum(np.multiply (prd-ytrain,prd-ytrain) )/ xtrain.shape[0]
	print "\n\nOverall accuracy ::\n"
	print "on training data ::"
	print accuracy_rmse(prd,ytrain)
	print "\n"
	print "on test data ::"        
    

visualize=False
feat=35
classes=3
alpha=0.001
K= 8
train=open("Assignment Classification/Set 1/Iris.tra")
test=open("Assignment Classification/Set 1/Iris.tes")
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



 
back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K)

        
    
    