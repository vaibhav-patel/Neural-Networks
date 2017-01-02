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

    
def back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K,spread):
    #prd=convert_to_output(feed_forward(xtrain,weights1,bias1,weights2,bias2))
    #print compute_error(prd,ytrain) 
    clusters= k_means(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K)

    new_dim=np.zeros((xtrain.shape[0],K))
    for i in range(0,epochs):        
        for i in range(K):
        	new_dim[:,i]=  activation(xtrain,clusters[i],spread)

        a1= np.dot(new_dim,np.transpose(weights1))+bias1
        prd=(a1)  
        delta2=(prd-ytrain)
        dW2= np.dot(np.transpose(delta2), new_dim )                
        """
        print a1[i]
        print h1[i]
        print a2[i]
        print a1.shape
        print h1.shape
        print prd.shape
        print weights2.shape
        print ytrain.shape
                
        print a2.shape
        
        raw_input()
        """
        weights1=weights1 - alpha*dW2
        bias1=bias1-alpha*np.sum(delta2)
        print np.sum(np.multiply (prd-ytrain,prd-ytrain) )/ xtrain.shape[0]
    print "\n\nRMSE error ::\n"    
    print "on training data ::"
    print np.sum(np.multiply (prd-ytrain,prd-ytrain) )/ xtrain.shape[0]
    print "\n"

    


visualize=False
feat=2
output=1
epochs=9000
alpha=0.001
spread=1
K=28
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
    


weights1=(np.random.rand(output,K)-.5)/1000
bias1=(np.random.rand(1,output)-.5)/1000



# feature normalization
# the hero, savier in this code
xtrain=featureNormalize(xtrain)
xval=featureNormalize(xval)


ytrain=np.reshape(ytrain,(ytrain.shape[0],1)) 
 
back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K,spread)

        
    
    