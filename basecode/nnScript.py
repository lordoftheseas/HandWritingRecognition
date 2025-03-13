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
    # σ(x) = 1/(1+exp(-x))
    z = np.array(z)
    z = 1 / (1 + np.exp(-z))

    return z # your code here

####################################################################################################################












####################################################################################################################

def preprocess():
    """Load and preprocess the MNIST dataset"""
    
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    
    #init empty arrays
    train_data = np.zeros((0, 784))
    train_label = np.zeros(0)
    validation_data = np.zeros((0, 784))
    validation_label = np.zeros(0)
    test_data = np.zeros((0, 784))
    test_label = np.zeros(0)
    
    #for tracking total examples
    train_count = 0
    
    #process each digit (0-9)
    for i in range(10):
        #get data for digit i
        digit_data = mat['train' + str(i)]
        
        #shuffle the data
        perm = np.random.permutation(digit_data.shape[0])
        digit_data = digit_data[perm]
        
        #determine split point for train/validation
        n_train = min(int(0.8 * digit_data.shape[0]), 50000 - train_count)
        
        #update count
        train_count += n_train
        
        #split into train/validation sets
        train_data = np.vstack((train_data, digit_data[:n_train]))
        train_label = np.append(train_label, np.ones(n_train) * i)
        
        validation_data = np.vstack((validation_data, digit_data[n_train:]))
        validation_label = np.append(validation_label, np.ones(digit_data.shape[0] - n_train) * i)
        
        #get test data for this digit
        test_digit = mat['test' + str(i)]
        test_data = np.vstack((test_data, test_digit))
        test_label = np.append(test_label, np.ones(test_digit.shape[0]) * i)
    
    #select features with non-zero std deviation
    std_dev = np.std(train_data, axis=0)
    selected_features = np.where(std_dev > 0)[0]
    
    #apply feature selection
    train_data = train_data[:, selected_features]
    validation_data = validation_data[:, selected_features]
    test_data = test_data[:, selected_features]
    
    #normalize to [0,1]
    train_data = train_data / 255.0
    validation_data = validation_data / 255.0
    test_data = test_data / 255.0
    
    print('preprocess done')
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label

####################################################################################################################












####################################################################################################################

def nnObjFunction(params, *args):
    """Compute neural network objective function and gradient"""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    #reshape params into weight matrices
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    n = training_data.shape[0]
    
    #add bias column to input
    input_data = np.hstack((training_data, np.ones((n, 1))))
    
    #forward pass
    #hidden layer
    z1 = sigmoid(np.dot(input_data, w1.T))
    z1 = np.hstack((z1, np.ones((n, 1))))  #add bias
    
    #output layer
    output = sigmoid(np.dot(z1, w2.T))
    
    #create target matrix
    y = np.zeros((n, n_class))
    for i in range(n):
        y[i, int(training_label[i])] = 1
    
    #calculate error
    error = -np.sum(y * np.log(output + 1e-10) + (1 - y) * np.log(1 - output + 1e-10)) / n
    
    #add regularization
    reg_term = (lambdaval / (2 * n)) * (np.sum(w1 * w1) + np.sum(w2 * w2))
    obj_val = error + reg_term
    
    #backpropagation
    #output layer gradient
    delta_output = output - y
    grad_w2 = np.dot(delta_output.T, z1) / n
    
    #hidden layer gradient
    delta_hidden = np.dot(delta_output, w2) * z1 * (1 - z1)
    delta_hidden = delta_hidden[:, :-1]  #remove bias
    grad_w1 = np.dot(delta_hidden.T, input_data) / n
    
    #add regularization to gradients
    grad_w1 += (lambdaval / n) * w1
    grad_w2 += (lambdaval / n) * w2
    
    #combine gradients
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    
    return (obj_val, obj_grad)

####################################################################################################################












####################################################################################################################

def nnPredict(w1, w2, data):
    """Predict class labels for data using trained weights"""
    
    #add bias term to input
    n = data.shape[0]
    data = np.hstack((data, np.ones((n, 1))))
    
    #forward pass
    #hidden layer
    hidden_out = sigmoid(np.dot(data, w1.T))
    hidden_out = np.hstack((hidden_out, np.ones((n, 1))))  #add bias
    
    #output layer
    output = sigmoid(np.dot(hidden_out, w2.T))
    
    #get predicted class
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

