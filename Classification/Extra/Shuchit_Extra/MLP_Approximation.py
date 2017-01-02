#!/usr/bin/python
import numpy as np
import pandas as pd
import math
import argparse

no_of_samples=0
input_features=0
output_features=0
input_data=None
output_data=None
hidden_neurons=0
learning_rate=0.01
no_of_epochs=10
input_weights=None
output_weights=None
train_file=None
test_file=None
activation_function_name=None
error_function_name=None

def check_args():
    # Parse args
    parser = argparse.ArgumentParser(description='MLP Approximation')
    parser.add_argument('train_file', metavar='Training_File', nargs=1, type=str,help='File for training',default='')
    parser.add_argument('-t','--test_file', metavar='Test_File', type=str,help='File for Test')
    parser.add_argument('-s','--save_folder', metavar='Save_Folder', type=str, help='Folder to save weights in',default='./') 
    parser.add_argument('-i','--input_features', metavar='Input_Features', type=int,help='Number of input features in data file default is No of columms - 1')
    parser.add_argument('-o','--output_features', metavar='Output_Features', type=int,help='Number of output features in data file default is 1')
    parser.add_argument('-n','--no_of_samples', metavar='No of Samples', type=int, help='Number of Samples to consider default all')
    parser.add_argument('-r','--ratio_train_validation', metavar='Ratio of Training samples to Validation samples', type=float, help='Ratio of training samples to validation samples. Eg: 0.90 for a 90-10 split. Defaults to 0.90', default=0.90)
    parser.add_argument('-H','--hidden_neurons', metavar='Hidden_Neurons', type=int, help='Number of hidden neurons to use defualt is 10',default=10)
    parser.add_argument('-e','--epochs', metavar='Epochs', type=int, help='Number of epochs to use defualt is 100',default=10)
    parser.add_argument('-l','--learning_rate', metavar='Learning_Rate', type=float, help='Learning rate to be used default 0.01',default=0.01)
    parser.add_argument('-d','--delimiter', metavar='Delimiter', type=str, help='Delimeter to use default tab',default="\s+")
    parser.add_argument('-E','--error-function', metavar='Error_Function', type=str, help='Error function to be used available choices are ',default='lsq',choices=['lsq','mlsq','hinge_loss','cross_entropy','fourth_power'])
    parser.add_argument('-a','--activation-function', metavar='Activation_Function', type=str, help='Activation function to be used available choices are ',default='sigmoid',choices=['sigmoid', 'bipolar_sigmoid', 'piece_wise_linear'])
    args = parser.parse_args()
    return args
    # print(args.accumulate(args.integers))

def initialize(parameters):
    global no_of_samples, input_features, output_features, input_data, output_data, hidden_neurons, learning_rate, no_of_epochs, input_weights, output_weights, test_file, train_file, activation_function_name, error_function_name

    # Reading train data
    train_file = parameters.train_file[0]
    if test_file is not None:
        test_file = parameters.test_file[0]
    delim = parameters.delimiter
    train_data = pd.read_csv(train_file,delimiter=delim,header=None)

    if parameters.no_of_samples is None:
        no_of_samples = train_data.shape[0]
    else:
        no_of_samples=parameters.no_of_samples

    # Setting the training-validation split
    no_of_samples=math.floor(no_of_samples*parameters.ratio_train_validation)

    if parameters.output_features is None:
        output_features = 1
    else:
        output_features = parameters.output_features
    
    if parameters.input_features is None:
        input_features = train_data.shape[1] - output_features
    else:
        input_features = parameters.input_features


    # np.random.shuffle(train_data.values[:,:])
    input_data = np.matrix(train_data.values[:,:-output_features])
    output_data = np.matrix(train_data.values[:,input_features:])

    # Initialize the Algorithm Parameters

    hidden_neurons = parameters.hidden_neurons
    learning_rate = parameters.learning_rate
    no_of_epochs = parameters.epochs
    activation_function_name = parameters.activation_function
    error_function_name = parameters.error_function

    # Initialize the weights

    input_weights = np.matrix(np.random.uniform(-1,1,size=(hidden_neurons,input_features)))
    output_weights = np.matrix(np.random.uniform(-1,1,size=(output_features,hidden_neurons)))

    # print(activation_function_name, error_function_name)

def activation_function(input_weights, input_sample, function_name):
    if function_name == 'sigmoid':
        return 1/(1+np.exp((-input_weights)*input_sample))
    elif function_name == 'bipolar_sigmoid': 
        return (2*activation_function(input_weights, input_sample, 'sigmoid')) - 1
    elif function_name == 'piece_wise_linear': 
        cut_off = 0.5
        x = np.squeeze(np.asarray(input_weights*input_sample))
        op =  np.piecewise(x,[x<-cut_off,np.logical_and(-cut_off<=x,x<cut_off),x>=cut_off],[0,lambda x : x+cut_off,1])
        op = np.matrix(op).T
        return op
    else:
        assert("Some error")
    
def activation_derivative(input_weights, input_sample, function_name):
    if function_name == 'sigmoid':
        func_value= activation_function(input_weights, input_sample, 'sigmoid')
        return np.multiply((func_value),(1 - func_value))
    elif function_name == 'bipolar_sigmoid': 
        func_value = activation_function(input_weights, input_sample, 'bipolar_sigmoid')
        return 0.5*np.multiply((1 - func_value),(1 + func_value))
    elif function_name == 'piece_wise_linear': 
        cut_off = 0.5
        x = np.squeeze(np.asarray(input_weights*input_sample))
        op = np.piecewise(x,[x<-cut_off,np.logical_and(-cut_off<=x,x<cut_off),x>=cut_off],[0,1,0])
        op = np.matrix(op).T
        return op
    else:
        assert("ERROR")
    
def error_function(output, target, function_name):
    if function_name == 'lsq':
        return 0.5*(np.multiply(target-output, target-output))
    elif function_name == 'mlsq':
        x = np.squeeze(np.asarray(target-output))
        op = np.piecewise(x,[x>= 1, x<1],[0, error_function(output, target, 'lsq')])
        op = np.matrix(op)
        return op
    elif function_name == 'fourth_power':
        return np.square(np.square(target-output))
    elif function_name == 'hinge_loss':
        pass
    elif function_name == 'cross_entropy':
        pass

def error_derivative(target, output, function_name):
    if function_name == 'lsq':
        return (target-output)
    elif function_name == 'mlsq':
        x = np.squeeze(np.asarray(target-output))
        op = np.piecewise(x,[x >= 1, x<1],[0, -error_derivative(output, target, 'lsq')])
        op = np.matrix(op)
        return op.T
    elif function_name == 'fourth_power':
        return 4*np.multiply(np.square(target-output),target-output)
    elif function_name == 'hinge_loss':
        pass
    elif function_name == 'cross_entropy':
        pass

def train():
    global no_of_samples, input_features, output_features, input_data, output_data, hidden_neurons, learning_rate, no_of_epochs, input_weights, output_weights, activation_function_name, error_function_name
    # Train the network
    for i in range(no_of_epochs):
        total_error=0
        delta_input_weights = np.zeros([hidden_neurons, input_features])
        delta_output_weights = np.zeros([output_features, hidden_neurons])
        for j in range(no_of_samples):
            input_sample = input_data[j, :]
            target = output_data[j, :]
            input_sample = input_sample.T
            target = target.T
            activ_func_value = activation_function(input_weights, input_sample, activation_function_name)
            output_hidden_layer = activ_func_value
            output_layer = output_weights*output_hidden_layer
            error_in_sample = target - output_layer
            err_deri_value = error_derivative(target, output_layer, error_function_name)
            activ_deri_value = activation_derivative(input_weights, input_sample, activation_function_name)
            err_func_value = error_function(output_layer, target, error_function_name)
            # print("-------------------------")
            # print("e_f_v:",err_func_value.T)
            # print("e_d_v:",err_deri_value.T)
            # print("a_f_v:",activ_func_value.T)
            # print("a_d_v:", activ_deri_value.T)
            # print("-------------------------")
            delta_output_weights = delta_output_weights + learning_rate * (err_deri_value*output_hidden_layer.T)
            alpha = np.multiply((output_weights.T * err_deri_value), activ_deri_value)
            delta_input_weights = delta_input_weights + learning_rate * (alpha * input_sample.T)
            total_error = total_error + err_func_value
            # print("total_error:",total_error)
        # print("input_weights:",input_weights.shape,"output_weights:",output_weights.shape)
        input_weights = input_weights + delta_input_weights
        output_weights = output_weights + delta_output_weights
    print("Training error:",math.sqrt(total_error/no_of_samples))
    return (input_weights, output_weights)

def validate():
    global no_of_samples, input_features, output_features, input_data, output_data, hidden_neurons, learning_rate, no_of_epochs, input_weights, output_weights, activation_function_name, error_function_name
    # Validate the network

    rms_error_validation = np.zeros([output_features, 1])
    result_validation = np.zeros([input_data.shape[0]-no_of_samples+1, 2])

    for j in range(no_of_samples,input_data.shape[0]):
        input_sample = input_data[j, :]
        target = output_data[j, :]
        input_sample = input_sample.T
        target = target.T
        activ_func_value = activation_function(input_weights, input_sample, activation_function_name)
        output_hidden_layer = activ_func_value
        output_layer=output_weights*output_hidden_layer
        error_in_sample = target - output_layer
        err_func_value = error_function(output_layer, target, error_function_name)
        rms_error_validation += err_func_value
        result_validation[j-no_of_samples] = [target,output_layer]

    print("Validation error:",math.sqrt(rms_error_validation/no_of_samples))

def test(test_file, weights,delim="\s+"):
    print("Testing Phase")
    # Test the network

    test_data = pd.read_csv(test_file, delimiter=delim,header=None)
    no_of_samples = test_data.shape[0]
    input_features = test_data.shape[1]
    input_data = np.matrix(test_data.values[:,:input_features])
    input_weights, output_weights = weights
    output = []

    for j in range(no_of_samples):
        input_sample = input_data[j, :]
        input_sample = input_sample.T
        activ_func_value = activation_function(input_weights, input_sample, activation_function_name)
        output_hidden_layer = activ_func_value
        output_layer=output_weights*output_hidden_layer
        output.append(output_layer)
        print(output_layer[0,0])

def verify_test(test_file, weights,delim="\s+"):
    global input_features, output_features
    # Test the network
    print("TEST PHASE")
    test_data = pd.read_csv(test_file, delimiter=delim,header=None)
    no_of_samples = test_data.shape[0]
    input_data = np.matrix(test_data.values[:,:input_features])
    output_data = np.matrix(test_data.values[:,input_features:])
    rms_error_testing = np.zeros([output_features,1])
    result_testing = np.zeros([no_of_samples,2])
    input_weights, output_weights = weights

    for j in range(no_of_samples):
        input_sample = input_data[j, :]
        input_sample = input_sample.T
        target = output_data[j,-1]
        activ_func_value = activation_function(input_weights, input_sample, activation_function_name)
        output_hidden_layer = activ_func_value
        output_layer=output_weights*output_hidden_layer
        error_in_sample = target - output_layer
        err_func_value = error_function(output_layer, target, error_function_name)
        rms_error_testing += err_func_value
        result_testing[j] = [target, output_layer]
        print(target,output_layer[0,0])
    
    print(math.sqrt(rms_error_testing/no_of_samples))


def main(parameters=None):
    if parameters is None:
        parameters = check_args()
    initialize(parameters)
    weights = train()

    # print(type(weights[0]),type(weights[1]))
    # s=str(parameters.train_file[0])+'_'+str(parameters.ratio_train_validation)+'_'+str(parameters.hidden_neurons)+'_'+str(parameters.epochs)+'_'+str(parameters.learning_rate)
    # s_input=s+'_input_weights.wts'
    # s_output=s+'_output_weights.wts'
    # print(s_input)
    # print(s_output)
    # print(weights[0].shape)
    # print(weights[1].shape)
    # np.savetxt(parameters.save_folder+s_input,weights[0])
    # np.savetxt(parameters.save_folder+s_output,weights[1])

    validate()
    if(parameters.test_file is not None):
        test(parameters.test_file,weights)

if __name__ == "__main__":
    main()
