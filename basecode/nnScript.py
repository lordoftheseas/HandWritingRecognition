import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W











####################################################################################################################

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1.0 / (1.0 + np.exp(-z))# your code here

####################################################################################################################












####################################################################################################################

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    train_data = np.zeros((0, 784)) #init empty arrays
    train_label = np.zeros(0)
    validation_data = np.zeros((0, 784))
    validation_label = np.zeros(0)
    test_data = np.zeros((0, 784))
    test_label = np.zeros(0)
    
    train_size = 0 #count train samples
    val_size = 0 #count val samples
    
    #process each digit (0-9)
    for i in range(10):
        digit_train = mat['train' + str(i)]
        n_samples = digit_train.shape[0]
        
        #shuffle data randomly
        idx = np.random.permutation(n_samples)
        digit_train = digit_train[idx]
        
        #calc split ratio to get ~50000 train examples
        train_idx = min(int(n_samples * 0.8), 50000 - train_size)
        
        #split into train/val
        train_portion = digit_train[:train_idx]
        val_portion = digit_train[train_idx:]
        
        #update counts
        train_size += train_portion.shape[0]
        val_size += val_portion.shape[0]
        
        #add to datasets
        train_data = np.vstack((train_data, train_portion))
        train_label = np.append(train_label, np.ones(train_portion.shape[0]) * i)
        
        validation_data = np.vstack((validation_data, val_portion))
        validation_label = np.append(validation_label, np.ones(val_portion.shape[0]) * i)
        
        #add test data
        test_digit = mat['test' + str(i)]
        test_data = np.vstack((test_data, test_digit))
        test_label = np.append(test_label, np.ones(test_digit.shape[0]) * i)

    # Feature selection
    # Your code here.
    #find non-repetitive features
    std_dev = np.std(train_data, axis=0)
    selected_features = np.where(std_dev > 0)[0]
    
    #apply feature selection
    train_data = train_data[:, selected_features]
    validation_data = validation_data[:, selected_features]
    test_data = test_data[:, selected_features]
    
    #normalize data to [0,1]
    train_data = train_data / 255.0
    validation_data = validation_data / 255.0
    test_data = test_data / 255.0

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

####################################################################################################################












####################################################################################################################

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    n = training_data.shape[0] #num of samples
    
    #add bias to input
    input_bias = np.hstack((training_data, np.ones((n, 1))))
    
    #forward pass: input->hidden
    a1 = np.dot(input_bias, w1.T) #calc linear combo
    z1 = sigmoid(a1) #apply activation
    
    #add bias to hidden layer
    z1_bias = np.hstack((z1, np.ones((n, 1))))
    
    #forward pass: hidden->output
    a2 = np.dot(z1_bias, w2.T) #calc linear combo
    output = sigmoid(a2) #apply activation
    
    #convert labels to one-hot
    y = np.zeros((n, n_class))
    for i in range(n):
        y[i, int(training_label[i])] = 1
    
    #calc error (neg log likelihood)
    error = -np.sum(y * np.log(output) + (1 - y) * np.log(1 - output)) / n
    
    #add regularization
    reg = (lambdaval / (2 * n)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    
    #total objective val
    obj_val = error + reg
    
    #backpropagation
    #output layer error
    delta2 = output - y
    
    #calc w2 gradient
    grad_w2 = np.dot(delta2.T, z1_bias) / n
    grad_w2 += (lambdaval / n) * w2 #add reg
    
    #hidden layer error
    delta1 = np.dot(delta2, w2) * (z1_bias * (1 - z1_bias))
    delta1 = delta1[:, :-1] #remove bias term gradient
    
    #calc w1 gradient
    grad_w1 = np.dot(delta1.T, input_bias) / n
    grad_w1 += (lambdaval / n) * w1 #add reg

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return (obj_val, obj_grad)

####################################################################################################################












####################################################################################################################

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    n = data.shape[0] #num of samples
    
    #add bias to input
    data_bias = np.hstack((data, np.ones((n, 1))))
    
    #forward pass: input->hidden
    hidden_in = np.dot(data_bias, w1.T)
    hidden_out = sigmoid(hidden_in)
    
    #add bias to hidden
    hidden_bias = np.hstack((hidden_out, np.ones((n, 1))))
    
    #forward pass: hidden->output
    output_in = np.dot(hidden_bias, w2.T)
    output = sigmoid(output_in)
    
    #get predicted class (max activation)
    labels = np.argmax(output, axis=1)

    return labels

####################################################################################################################












"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

