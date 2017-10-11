# DL实验大纲

## exp_1_2_1_numpy 基础
1. Building basic functions with numpy
    1. Sigmoid function, np.exp()
    2. Sigmoid gradient
    3. Reshaping arrays
    4. Normalizing rows
    5. Broadcasting and the softmax function
2. Vectorization
    1. Implement the L1 and L2 loss functions

## exp_1_2_2_Logistic Regression
1. Packages
2. Overview of the Problem set
3. General Architecture of the learning algorithm
4. Building the parts of our algorithm
    1. Helper functions
    2. Initializing parameters
    3. Forward and Backward propagation
5. Merge all functions into a model
6. Further analysis
7. Test with your own image


## exp_1_3_Planar data classification with a hidden layer
1. Packages
2. Dataset
3. Simple Logistic Regression
4. Neural Network model
    1. Defining the neural network structure
    2. Initialize the model's parameters
    3. The Loop
    4. Integrate parts 4.1, 4.2 and 4.3 in nn_model()
    5. Predictions
    6. Tuning hidden layer size 
5. Performance on other datasets

## exp_1_4_1_Building your Deep Neural Network Step by Step
1. Packages
2. Outline of the Assignment
3. Initialization
    1. 2-layer Neural Network
    2. L-layer Neural Network
4. Forward propagation module
    1. Linear Forward
    2. Linear-Activation Forward
5. Cost function
6. Backward propagation module
    1. Linear backward
    2. Linear-Activation backward
    3. L-Model Backward
    4. Update Parameters
7. Conclusion

## exp_1_4_2_Deep Neural NetWork - Application
1. Packages
2. Dataset
3. Architecture of your model
    1. 2-layer neural network
    2. L-layer deep neural network
    3. General methodology
4. Two-layer neural network
5. L-layer Neural Network
6. Results Analysis
7. Test with your own image 

## exp_2_1_1_初始化（Initialization）
1. Neural Network model
2. Zero initialization
3. Random initialization
4. He initialization
5. Conclusions

## exp_2_1_2_正则化 （Regularization）
1. Non-regularized model
2. L2 Regularization
3. Dropout
    1. Forward propagation with dropout
    2. Backward propagation with dropout
4. Conclusions

## exp_2_1_3_梯度检查（Gradient Checking）
1. How does gradient checking work?
2. 1-dimensional gradient checking
3. N-dimensional gradient checking

## exp_2_2_最优化(Optimization)
1. Gradient Descent
2. Mini-Batch Gradient descent
3. Momentum
4. Adam
5. Model with different optimization algorithms
    1. Mini-batch Gradient descent
    2. Mini-batch gradient descent with momentum
    3. Mini-batch with Adam mode
    4. Summary

##  exp_2_3_框架TensorFlow
1. Exploring the Tensorflow Library
    1. Linear function
    2. Computing the sigmoid
    3. Computing the Cost
    4. Using One Hot encodings
    5. Initialize with zeros and ones
2. Building your first neural network in tensorflow

    0. Problem statement: SIGNS Dataset
    1. Create placeholders
    2. Initializing the parameters
    3. Forward propagation in tensorflow
    4. Compute cost
    5. Backward propagation & parameter updates
    6. Building the model
    7. Test with your own image