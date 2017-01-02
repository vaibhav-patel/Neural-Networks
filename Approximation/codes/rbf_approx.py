# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 01:57:47 2016

@author: Vaibhav Amit Patel
"""

def featureNormalize(X):
    X_norm = np.divide((X-np.mean(X,axis=0)), np.std(X,axis=0))
    
    return X_norm

def activation(a,mu,spread):
    #here it is RBF function
    a=np.exp(-np.sum(np.multiply((a-mu),(a-mu)),axis=1)/(2*spread*spread))
    return a


def rbf(a,mu,spread):
    #here it is RBF function
    a=np.exp(-np.sum(np.multiply((a-mu),(a-mu)),axis=1)/(2*spread*spread))
    return a

def feed_forward(xtrain,weights1,bias1,weights2,bias2):
    a1= np.dot(xtrain,np.transpose(weights1))+bias1
    h1=activation(a1)        
    a2= np.dot(h1,np.transpose(weights2))+bias2
    prd=(a2)  
    return prd

    
    


def accuracy_rmse(xtrain,weights1,bias1,weights2,bias2,y):
    prd=(feed_forward(xtrain,weights1,bias1,weights2,bias2))    
    return float(100*np.sum(prd==y))/y.shape[0]
    
def k_means(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K):
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
	while err>=0.05:	
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

    
def back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K,spread):
    clusters= k_means(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K)
    #clusters= xtrain[np.random.choice(xtrain.shape[0], size=K,replace=False)]
    new_dim=np.zeros((xtrain.shape[0],K))
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
    weights1=answer[0:answer.shape[0]-1, :]
    bias1=answer[-1,:]
    weights1=weights1.T
    bias1=bias1.T
    a1= np.dot(saved_dim,np.transpose(weights1))+bias1
    prd=(a1)
		

    import matplotlib.pyplot as plt 
    
    plt.plot(range(ytrain.shape[0]),ytrain)
    plt.plot(range(prd.shape[0]),prd)
    plt.title('pseudo inv Training Actual function vs true function for '+ str(K)+' RBF centers') 
    plt.legend(('True function','Approximated function function',),loc='upper left')

    plt.show()

    new_dim=np.zeros((xtest.shape[0],K))
    for i in range(K):
        new_dim[:,i]=  rbf(xtest,clusters[i],spread)
    a1= np.dot(new_dim,np.transpose(weights1))+bias1
    tprd=(a1)
	

    print "\n\nRMSE error ::\n"    
    print "on training data ::"
    print np.power((np.sum(np.multiply (prd-ytrain,prd-ytrain) )/ xtrain.shape[0]) ,0.5)
    print "\n"
    
    print "\n\nRMSE error ::\n"    
    print "on testing data ::"
    print np.power((np.sum(np.multiply (tprd-ytest,tprd-ytest) )/ xtest.shape[0]) ,0.5)
    print "\n"
    plt.plot(range(ytrain.shape[0]),ytrain)
    plt.plot(range(prd.shape[0]),prd)
    plt.title('pseudo inv Testing Actual function vs true function for '+ str(K)+' RBF centers') 
    plt.legend(('True function','Approximated function function',),loc='upper left')

    plt.show()



visualize=False
feat=2
output=1
epochs=400
alpha=0.0001
spread=.5
K=40
train=open("approximationproblems/bj.tra")
test=open("approximationproblems/bj.tes")
ytest=open("approximationproblems/bj.y")

train=train.read()
test=test.read()
ytest=ytest.read()


train = [float(x) for x in train.split()]
test = [float(x) for x in test.split()]
ytest = [float(x) for x in ytest.split()]

import numpy as np
train=np.array(train)
test=np.array(test)
ytest=np.array(ytest)


def change_shape(x,f):    
    m=x.shape[0]
    return np.reshape(x,(m/f,f))

train=change_shape(train,feat+1)
test=change_shape(test,feat)


xtrain=train[:,0:feat]
ytrain=train[:,feat]
xtest=test[:,0:feat]

total_train=xtrain.shape[0]
total_val=int(total_train*0.1)

#10-fold validation
xval=xtrain[total_train-total_val:total_train,:]
xtrain=xtrain[0:total_train-total_val,:]
yval=ytrain[total_train-total_val:total_train]
ytrain=ytrain[0:total_train-total_val]
    


weights1=(np.random.rand(output,K)-.5)/1000
bias1=(np.random.rand(1,output)-.5)/1000



# feature normalization
# the hero, savier in this code
xtrain=featureNormalize(xtrain)
xval=featureNormalize(xval)
xtest=featureNormalize(xtest)

ytrain=np.reshape(ytrain,(ytrain.shape[0],1)) 
yval=np.reshape(yval,(yval.shape[0],1)) 
ytest=np.reshape(ytest,(ytest.shape[0],1)) 
 
back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K,spread)

        
    
    