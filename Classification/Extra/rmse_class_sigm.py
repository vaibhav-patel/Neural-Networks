
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
    prd=activation(a2)  
    return prd

def convert_to_output(prd):
    prd=np.argmax(prd,axis=1)
    return prd


def confusion_matrix(prd,y):
    cm=np.zeros((classes,classes))
    for idx,i in enumerate(y):
        cm[i][prd[idx]]+=1
    return cm    



    
    
def compute_error(xtrain,weights1,bias1,weights2,bias2,y):
    # here this is mean square error
    prd=(feed_forward(xtrain,weights1,bias1,weights2,bias2))
    y=convert_to_output(y)
    prd=convert_to_output(prd)
    return  ((float(np.sum(np.multiply((prd-y),(prd-y))))/(2*y.shape[0])))
    


def diff_act(a):
    #here sigmoid fucntion
    return np.multiply(activation(a),1-activation(a))




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
    #prd=convert_to_output(feed_forward(xtrain,weights1,bias1,weights2,bias2))
    #print compute_error(prd,ytrain) 
    for i in range(0,epochs):
        a1= np.dot(xtrain,np.transpose(weights1))+bias1
        h1=activation(a1)        
        a2= np.dot(h1,np.transpose(weights2))+bias2
        prd=activation(a2)
        delta2=np.multiply(np.multiply((prd-ytrain),prd),1- prd )
        dW2= np.dot(np.transpose(delta2),h1)  
        weights2=weights2 - alpha*dW2
        bias2=bias2-alpha*np.sum(delta2)
        delta1=(np.multiply(np.multiply(np.dot(delta2,weights2),h1),(1-h1)))        
        dW1= np.dot(np.transpose(delta1),xtrain)                
        weights1=weights1 - alpha*dW1
        bias1=bias1-alpha*np.sum(dW1)
        print compute_error(xtrain,weights1,bias1,weights2,bias2,ytrain)

    ta1= np.dot(test,np.transpose(weights1))+bias1
    th1=activation(ta1)        
    ta2= np.dot(th1,np.transpose(weights2))+bias2
    tprd=activation(ta2)
    prd2=prd
    ytrain2=ytrain    
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


feat=34
classes=2
epochs=1000
alpha=0.001
hidden=100
train=open("ION.tra")
test=open("ION.tes")
train=train.read()
test=test.read()
answer=open("ION.cla")
answer=answer.read()


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

np.random.seed(25)
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


ytrain=one_hot(ytrain)
yval=one_hot(yval)
answer=one_hot(answer)

# feature normalization
# the hero, savier in this code
xtrain=featureNormalize(xtrain)
xval=featureNormalize(xval)



    
back_prop(xtrain,weights1,bias1,weights2,bias2,ytrain,alpha,yval,xval)

        
    
    
