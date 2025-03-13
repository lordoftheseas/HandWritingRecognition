import numpy as np
import sys
import os
from scipy.io import loadmat

# Try to import functions from nnScript
try:
    from nnScript import sigmoid, preprocess, nnObjFunction, nnPredict, initializeWeights
    print("Successfully imported functions from nnScript.py")
except ImportError:
    print("Error: Could not import functions from nnScript.py")
    print("Make sure nnScript.py is in the same directory as this file")
    sys.exit(1)

def test_sigmoid():
    """Test the sigmoid function with various inputs"""
    print("\n==== Testing sigmoid() function ====")
    
    # Test with scalar
    x_scalar = 0
    result_scalar = sigmoid(x_scalar)
    expected_scalar = 0.5
    print(f"sigmoid(0) = {result_scalar}, Expected: {expected_scalar}")
    if abs(result_scalar - expected_scalar) < 0.0001:
        print("✓ Scalar test passed")
    else:
        print("✗ Scalar test failed")
    
    # Test with vector
    x_vector = np.array([-5, -1, 0, 1, 5])
    result_vector = sigmoid(x_vector)
    expected_vector = np.array([0.0066928, 0.26894142, 0.5, 0.73105858, 0.9933072])
    print(f"sigmoid(vector) = {result_vector}")
    print(f"Expected: {expected_vector}")
    max_diff = np.abs(result_vector - expected_vector).max()
    print(f"Maximum difference: {max_diff}")
    if max_diff < 0.0001:
        print("✓ Vector test passed")
    else:
        print("✗ Vector test failed")
    
    # Test with matrix
    x_matrix = np.array([[-1, 0], [1, 2]])
    result_matrix = sigmoid(x_matrix)
    expected_matrix = np.array([[0.26894142, 0.5], [0.73105858, 0.88079708]])
    print(f"sigmoid(matrix) = \n{result_matrix}")
    print(f"Expected: \n{expected_matrix}")
    max_diff_matrix = np.abs(result_matrix - expected_matrix).max()
    print(f"Maximum difference: {max_diff_matrix}")
    if max_diff_matrix < 0.0001:
        print("✓ Matrix test passed")
    else:
        print("✗ Matrix test failed")
    
    return max_diff < 0.0001 and max_diff_matrix < 0.0001

def test_preprocess():
    """Test the preprocess function"""
    print("\n==== Testing preprocess() function ====")
    
    # Check if mnist_all.mat exists
    if not os.path.exists('mnist_all.mat'):
        print("Error: mnist_all.mat not found in the current directory")
        print("Download the file and place it in the same directory as this script")
        return False
    
    # Try to load and preprocess data
    try:
        # Call preprocess - use unpacking based on your implementation
        try:
            train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features = preprocess()
            print("Got 7 return values from preprocess")
        except ValueError:
            # Try with 6 return values
            train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
            selected_features = None
            print("Got 6 return values from preprocess")
        
        # Basic checks
        print(f"Training data shape: {train_data.shape}")
        print(f"Training label shape: {train_label.shape}")
        print(f"Validation data shape: {validation_data.shape}")
        print(f"Validation label shape: {validation_label.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Test label shape: {test_label.shape}")
        
        # Check if shapes seem reasonable
        if train_data.shape[0] > 0 and validation_data.shape[0] > 0 and test_data.shape[0] > 0:
            print("✓ Data shapes look reasonable")
        else:
            print("✗ Some data shapes have 0 samples")
            return False
        
        # Check if data is normalized
        if 0 <= train_data.min() and train_data.max() <= 1:
            print("✓ Training data is normalized [0,1]")
        else:
            print(f"✗ Training data range: [{train_data.min()}, {train_data.max()}]")
            print("  Data should be normalized to [0,1]")
            return False
        
        # Check feature selection if available
        if selected_features is not None:
            print(f"Number of selected features: {len(selected_features)}")
            if len(selected_features) > 0:
                print("✓ Feature selection seems to work")
            else:
                print("✗ No features were selected")
                return False
            
        # Check one-hot encoding of labels
        sample_labels = train_label[:5]
        print(f"Sample labels (first 5 rows):\n{sample_labels}")
        
        # Check if labels are one-hot encoded
        if train_label.shape[1] == 10 and np.all(np.sum(train_label, axis=1) == 1):
            print("✓ Labels are one-hot encoded")
        else:
            print("✗ Labels don't appear to be one-hot encoded")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error in preprocess function: {e}")
        return False

def test_nnObjFunction():
    """Test the neural network objective function"""
    print("\n==== Testing nnObjFunction() function ====")
    
    # Create small synthetic dataset
    np.random.seed(42)  # For reproducibility
    n_samples = 10
    n_features = 3
    n_hidden = 4
    n_class = 2
    
    # Generate data
    data = np.random.rand(n_samples, n_features)
    labels = np.zeros((n_samples, n_class))
    for i in range(n_samples):
        labels[i, np.random.randint(0, n_class)] = 1
    
    # Initialize weights
    w1 = initializeWeights(n_features, n_hidden)
    w2 = initializeWeights(n_hidden, n_class)
    params = np.concatenate((w1.flatten(), w2.flatten()), 0)
    
    # Run objective function without regularization
    args1 = (n_features, n_hidden, n_class, data, labels, 0)
    try:
        obj_val1, obj_grad1 = nnObjFunction(params, *args1)
        print(f"Objective value (λ=0): {obj_val1}")
        print(f"Gradient shape: {obj_grad1.shape}")
        
        # Run with regularization
        args2 = (n_features, n_hidden, n_class, data, labels, 1.0)
        obj_val2, obj_grad2 = nnObjFunction(params, *args2)
        print(f"Objective value (λ=1.0): {obj_val2}")
        
        # Basic checks
        expected_grad_shape = w1.size + w2.size
        if obj_grad1.shape[0] == expected_grad_shape:
            print("✓ Gradient has correct shape")
        else:
            print(f"✗ Gradient shape: {obj_grad1.shape}, Expected: {expected_grad_shape}")
            return False
        
        # Check if regularization increases objective value
        if obj_val2 > obj_val1:
            print("✓ Regularization increases objective value")
        else:
            print(f"✗ Regularization should increase objective value")
            print(f"  Value without regularization: {obj_val1}")
            print(f"  Value with regularization: {obj_val2}")
            return False
        
        # Check gradient differences with and without regularization
        grad_diff = np.abs(obj_grad2 - obj_grad1).mean()
        print(f"Average gradient difference with regularization: {grad_diff}")
        if grad_diff > 0:
            print("✓ Regularization affects gradients")
        else:
            print("✗ Regularization doesn't change gradients")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error in nnObjFunction: {e}")
        return False

def test_nnPredict():
    """Test the neural network prediction function"""
    print("\n==== Testing nnPredict() function ====")
    
    # Create small synthetic dataset
    np.random.seed(42)
    n_samples = 5
    n_features = 3
    n_hidden = 2
    n_class = 4
    
    # Generate data
    data = np.random.rand(n_samples, n_features)
    
    # Create predictable weights
    w1 = np.array([[0.1, 0.2, 0.3, 0.4], 
                   [0.5, 0.6, 0.7, 0.8]])  # (2, 4) - hidden units x (features + 1)
    w2 = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6],
                   [0.7, 0.8, 0.9],
                   [0.2, 0.3, 0.4]])  # (4, 3) - classes x (hidden units + 1)
    
    try:
        # Make predictions
        labels = nnPredict(w1, w2, data)
        print(f"Predictions: {labels}")
        
        # Manually compute prediction for first sample
        x = data[0]
        print(f"\nManual verification for first sample:")
        print(f"Input: {x}")
        
        # Hidden layer
        x_bias = np.append(x, 1)
        z = sigmoid(np.dot(w1, x_bias))
        z_bias = np.append(z, 1)
        print(f"Hidden layer output (with bias): {z_bias}")
        
        # Output layer
        o = sigmoid(np.dot(w2, z_bias))
        print(f"Output layer: {o}")
        
        # Get prediction
        pred = np.argmax(o)
        print(f"Manual prediction for first sample: {pred}")
        print(f"Function prediction for first sample: {labels[0]}")
        
        if pred == labels[0]:
            print("✓ Manual calculation matches function output")
        else:
            print("✗ Manual calculation doesn't match function output")
            return False
        
        # Check prediction shape
        if len(labels) == n_samples:
            print("✓ Prediction has correct shape")
        else:
            print(f"✗ Prediction shape: {labels.shape}, Expected: ({n_samples},)")
            return False
        
        # Check if all predictions are valid classes
        if np.all(labels >= 0) and np.all(labels < n_class):
            print("✓ All predictions are valid classes")
        else:
            print("✗ Some predictions are invalid classes")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error in nnPredict: {e}")
        return False

def test_all_integrated():
    """Test all functions together in an integrated way"""
    print("\n==== Integrated Test with Small Dataset ====")
    
    try:
        # Create small synthetic dataset
        np.random.seed(42)
        n_samples = 20
        n_features = 5
        n_hidden = 3
        n_class = 4
        
        # Generate data
        X = np.random.rand(n_samples, n_features)
        y = np.zeros((n_samples, n_class))
        for i in range(n_samples):
            y[i, np.random.randint(0, n_class)] = 1
        
        # Split into train/test
        train_data = X[:15]
        train_label = y[:15]
        test_data = X[15:]
        test_label = y[15:]
        
        # Initialize weights
        initial_w1 = initializeWeights(n_features, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
        
        # Compute objective function
        lambdaval = 0.1
        args = (n_features, n_hidden, n_class, train_data, train_label, lambdaval)
        obj_val, obj_grad = nnObjFunction(initialWeights, *args)
        print(f"Initial objective value: {obj_val}")
        
        # Make predictions with initial weights
        pred_train = nnPredict(initial_w1, initial_w2, train_data)
        train_acc = 100 * np.mean((pred_train == np.argmax(train_label, axis=1)).astype(float))
        
        pred_test = nnPredict(initial_w1, initial_w2, test_data)
        test_acc = 100 * np.mean((pred_test == np.argmax(test_label, axis=1)).astype(float))
        
        print(f"Training accuracy with initial weights: {train_acc}%")
        print(f"Test accuracy with initial weights: {test_acc}%")
        
        print("✓ Integrated test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in integrated test: {e}")
        return False

def run_all_tests():
    """Run all tests and report overall results"""
    results = {}
    
    print("====== Neural Network Function Tests ======")
    print("Make sure nnScript.py and mnist_all.mat are in the same directory")
    
    # Run all tests
    results['sigmoid'] = test_sigmoid()
    results['preprocess'] = test_preprocess()
    results['nnObjFunction'] = test_nnObjFunction()
    results['nnPredict'] = test_nnPredict()
    results['integrated'] = test_all_integrated()
    
    # Print summary
    print("\n====== Test Summary ======")
    for test, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test}: {status}")
    
    # Overall result
    if all(results.values()):
        print("\n🎉 All tests passed! Your neural network implementation looks good.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    run_all_tests()