# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:20:20 2016

@author: vaibh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:15:51 2016

@author: vaibh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:12:58 2016

@author: vaibh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:09:31 2016

@author: vaibh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:59:19 2016

@author: vaibh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:51:34 2016

@author: vaibh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:51:08 2016

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

def rbf(a,mu,spread):
    #here it is RBF function
    a=np.exp(-np.sum(np.multiply((a-mu),(a-mu)),axis=1) /  (2*spread*spread))
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

  
    

def accuracy_rmse(aprd,ay):
    ay=convert_to_output(ay)
    aprd=convert_to_output(aprd)
    #print "confusion matrix :: "
    #print confusion_matrix(aprd,ay)
    
    return float(100*np.sum(aprd==ay))/ay.shape[0]
    
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
    
def accuracy_average(prd,y):
    y=convert_to_output(y)
    prd=convert_to_output(prd)
    cls=np.zeros((classes,1))
    for i in range(classes):
        cls[i]=np.sum(y==i)
    acc=0
    cnf=confusion_matrix(prd,y)
    for i in range(classes):
		if(cls[i] !=  0):
			acc= acc + ( cnf[i,i]/float(cls[i]))
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
	while True:	
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
		if(np.absolute(olderr-err) <= threshKmeans):
			break;
		olderr=err

	return [clusters,olderr]
    
def back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K,clusters,test):
    clusters= xtrain[np.random.choice(xtrain.shape[0], size=K,replace=False)]
    spread=np.ones((K,1))
    #clusters=k_means(xtrain,K)[0]
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
    tr_errs=np.zeros((epochs,1))
    val_errs=np.zeros((epochs,1))
    overfit_i=epochs;
    for i in range(epochs):
        for j in range(K):
            new_dim[:,j]=  rbf(xtrain,clusters[j],spread[j])
        a1= np.dot(new_dim,np.transpose(weights1))+bias1
        prd=a1                
        delta2=prd-ytrain       
        dW2= np.dot(np.transpose(delta2), new_dim )
        delta_center=np.zeros((K,xtrain.shape[1]))
        for j in range(K):
            rp=np.repeat (new_dim[:,j],new_dim[:,j].shape[0], axis=0)
            rp=np.reshape(rp,(xtrain.shape[0],xtrain.shape[0]))
            delta_center[j] = np.dot (weights1.T[j,:], (np.dot( np.dot(delta2.T,  rp.T) , (xtrain-clusters[j])/(2*spread[j]*spread[j]))) )
    
        weights1=weights1 - alpha*dW2
        bias1=bias1-alpha*np.sum(delta2)
        clusters=clusters + 2*alpha_c*(delta_center)
        if visualize2:
            plt.pause(0.05)            
            cnt=cnt+1
            plt.plot(cnt,spread[0],'x')
            plt.draw()

        arr=np.zeros((xtrain.shape[0],1))
        arrv=np.zeros((xtrain.shape[0],K))
        ds=np.zeros((K,1))
        for j in range(K):
            a1= np.multiply(new_dim[:,j], np.sum(np.power((xtrain-clusters[j]),2),axis=1))
            a2= np.sum(np.dot(delta2,weights1),axis=1)
            a3=np.multiply(a1,a2)
            ds[j]=np.sum(a3)
            #print ds[j]
            ds[j]=(ds[j]*(-2))/(spread[j]*spread[j]*spread[j])
            #print ds[j]
            spread[j]=spread[j]-alpha_spread*ds[j]
            #print spread[j]
            #raw_input()

        #print spread
        #raw_input()
        #print spread
        print i
        if visualize:
	        plt.pause(0.05)
	        plt.plot(clusters[:,0],clusters[:,1],'x')
	        plt.draw()
        tr_errs[i]=np.sum(np.multiply (prd-ytrain,prd-ytrain) )/ xtrain.shape[0]
        #print tr_errs[i]
        #after every 10 epochs check validation error
        if(i%1==0):
            v_new_dim=np.zeros((xval.shape[0],K))
            for j in range(K):
                v_new_dim[:,j]=  rbf(xval,clusters[j],spread[j])
            val_a1= np.dot(v_new_dim,np.transpose(weights1))+bias1
            val_prd=val_a1
            val_errs[i]=np.sum(np.multiply (val_prd-yval,val_prd-yval) )/ xval.shape[0]
            #if(accuracy_rmse(prd,ytrain) -accuracy_rmse(val_prd,yval) > 8  and i>30):
             #   overfit_i=i
                #break;    
    #import matplotlib.pyplot as plt
    print overfit_i
    #plt.plot(range( overfit_i), tr_errs[0:overfit_i], 'o')	
    #plt.show()	
    #plt.plot(range(overfit_i), val_errs[0:overfit_i], 'o')	
    #plt.show()	
    
    v_new_dim=np.zeros((xval.shape[0],K))      
    for i in range(K):
        v_new_dim[:,i]=  rbf(xval,clusters[i],spread[j])
    val_a1= np.dot(v_new_dim,np.transpose(weights1))+bias1
    val_prd=val_a1
    
    print "\n\nOverall accuracy ::\n"    
    #print "on training data ::"
    #print accuracy_rmse(prd,ytrain)
    #print "\n"
    #print "on validation data ::"  
    #print accuracy_rmse(val_prd,yval)
    #print "\n\nGeometric mean accuracy ::\n"    
    #print "on training data ::"
    #print accuracy_gm(prd,ytrain)
    #print "\n"    
    #print "on validation data ::"        
    #print accuracy_gm(val_prd,yval)
    #print "\n"
    
    #rint "\n\nAverage mean accuracy ::\n"    
    #print "on training data ::"
    #print accuracy_average(prd,ytrain)
    #print "\n"    
    #print "on validation data ::"        
    #print accuracy_average(val_prd,yval)
    #print "\n"
    import xlwt
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    sheet1.write(0, 0, K)
    sheet1.write(1,0, overfit_i)
    
    prd=np.concatenate((prd,val_prd))
    ytrain=np.concatenate((ytrain,yval))
    
    print accuracy_rmse(prd,ytrain)
    sheet1.write(1,1,accuracy_rmse(prd,ytrain))
    sheet1.write(1,2,float(accuracy_gm(prd,ytrain)))
    sheet1.write(1,3,float(accuracy_average(prd,ytrain)))
    
    prd=convert_to_output(prd)
    ytrain=convert_to_output(ytrain)
    
    cnf= confusion_matrix(prd,ytrain)
    print np.sum(cnf)
    for v1 in range(classes):
        for v2 in range(classes):
            sheet1.write(6+v1,6+v2,float(cnf[v1][v2] ) )
    
    v_new_dim=np.zeros((test.shape[0],K))      
    for i in range(K):
        v_new_dim[:,i]=  rbf(test,clusters[i],spread[j])
    val_a1= np.dot(v_new_dim,np.transpose(weights1))+bias1
    val_prd=val_a1
    print float(accuracy_rmse(val_prd,answer))
    sheet1.write(2,1,float(accuracy_rmse(val_prd,answer)))
    sheet1.write(2,2,float(accuracy_gm(val_prd,answer)))
    sheet1.write(2,3,float(accuracy_average(val_prd,answer)))
    
    val_prd=convert_to_output(val_prd)
    answerv=convert_to_output(answer)
    
    cnf= confusion_matrix(val_prd,answerv)
    print np.sum(cnf)
    for v1 in range(classes):
        for v2 in range(classes):
            sheet1.write(12+v1,12+v2,float(cnf[v1][v2] ) )
    
    
    for i in range(0,prd.shape[0]):
        #print i
        
        sheet1.write(i+3,0, float(prd[i]) )
    for i in range(0,val_prd.shape[0]):
        sheet1.write(i+3,1, float(val_prd[i]) )
    

    book.save(""+name+"/a"+str(folder)+"  "+str(K)+"  "+str(overfit_i)+".xls")
    #for vap8 in range(ex1.shape[0]):
    return [weights1, bias1]

	
name='Wine'
folder=10
# threshold for 
threshKmeans=0.001
threshfindK=0.03


visualize2=False


visualize=False

#number of features
feat=13

#number of classes
classes=3

#maximum epochs to run (provided there is no significant overfitting)
epochs=50

# learning rate(change it to adaptive learning rate)
alpha=0.001

#learning rate for cluster  update
alpha_c=0.0001

#learning rate for updating spread 
alpha_spread=0.000001

# order(which will be intitalized below)
#maximum hidden neurons's we want
order= feat*classes
K=15


train=open("Assignment Classification/Set "+str(folder)+"/"+name+".tra")
test=open("Assignment Classification/Set "+str(folder)+"/" +name+ ".tes")
answer=open("Assignment Classification/Results/Group "+str(folder)+"/" +name+ ".cla")
answer=answer.read()

train=train.read()
test=test.read()


train = [float(x) for x in train.split()]
test = [float(x) for x in test.split()]
answer = [float(x) for x in answer.split()]

import numpy as np
train=np.array(train)
test=np.array(test)
answer=np.array(answer)


def change_shape(x,f):    
    m=x.shape[0]
    return np.reshape(x,(m/f,f))

train=change_shape(train,feat+1)
test=change_shape(test,feat)
ex1=train[:,0:train.shape[1]-1]
aex1=train[:,train.shape[1]-1]
ex2=test


#total examples
total_train=train.shape[0]

#fold size
fsize=total_train/10

#if the training data is not a multiple of 10 then there will be some data loss

#create folds of the data
folds_10=np.zeros((10,fsize,train.shape[1]))


for i in range(0,10):
	folds_10[i]=train[(i)*fsize:(i+1)*fsize,:]


fin_weights1=np.zeros((classes,K))
fin_bias1=np.zeros((1,classes))

def one_hot(y):
	    m=y.shape[0]
	    narr=np.ones((m,classes))*-1
	    narr=np.zeros((m,classes))

	    y=y.astype(int)    
	    for idx,i in enumerate(y):
	        narr[idx][i-1]=1
	    return narr
aex1=one_hot(aex1)

answer=one_hot(answer)
tot=train.shape[0]

xtrain=train[ 0:int(tot*0.9 ),0:train.shape[1]-1]
ytrain=train[0:int(tot*0.9 ),train.shape[1]-1]

xval=train[int(tot*0.9 ):tot, 0:train.shape[1]-1]
yval=train[int(tot*0.9 ):tot,train.shape[1]-1]

ytrain=one_hot(ytrain)
yval=one_hot(yval) 
 

xtrain=featureNormalize(xtrain)
xval=featureNormalize(xval)
test=featureNormalize(test)
np.random.seed(12)
weights1=(np.random.rand(classes,K)-.5)/1000
bias1=(np.random.rand(1,classes)-.5)/1000

#for vp in range(classes,order,3):
clusters =k_means(xtrain,K)[0]
	
ls= back_prop(xtrain,weights1,bias1,ytrain,alpha,yval,xval,K,clusters,test)
