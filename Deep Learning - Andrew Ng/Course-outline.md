'Deep Learning 课程大纲'

# 课程1 - Neural Networks and Deep Learning

## Week1 - Introduction to deep learning
1. Welcome to the Deep Learning Specialization
    * 整个课程的结构
2. Introduction to Deep Learning
    * what is a neural network?
        - ReLu = Rectified Linear Units（线性整流函数）
    * supervised Learning with Neural Networks
        - 各个领域的运用
        - 结构化数据（数据库数据）/非结构化数据（音频、图像、文字）
    * Why is Deep Learning taking off?
        -   图像： performance / 数据量- 
    * About this Course
    * Course Respurces
3. 习题测试
4. 采访 - Geoffrey Hinton

## Week2 - Neural Networks Basics
1. Logistic Regression as a Neural Network
    * Binary Classification
    * Logistic Regression
    * Logistic Regression Cost Function
        - loss function | cost function
    * Gradient Decent
    * Derivatives
    * More Derivative Example
    * Computation Graph
    * Derivatives with a computation Graph
    * Logistic Regression Gradient Descent
    * Gradient Descent on m example
2. Python and Vectorization
    * Vectorization (避免 for LOOP)
    * More Vectorization Examples （利用 numpy）
    * Vectorizing Logistic Regression （去掉 m-for）
    * Vectorizing Logistic Regression's Gradient Output （去掉 n-for）
    * Broadcasting in Python 
    * A note on python/numpy vectors(不要用 rank 1 vector)
    * Quick Tour of Jupyter/iPython Notebooks
    * Explanation of logistic Regression cost function(最大似然)
3. 习题测试
4. programming Assignments
    * Python Basics with numpy（python.numpy 基础）
    * Logistic Regression with a Neural Network mindset
5. 采访 Pieter Abbeel interview

## Week3 - Shallow neural networks
1. Shallow neural networks
    * Neural Networks Overview
    * Neural Network Representation
    * Computing a Neural Network's Output
    * 多样本向量化
    * 向量化实现的解释
    * 激活函数 sigmoid | tanh| ReLU
    * 为什么需要激活函数
    * 激活函数的导数
    * 神经网络的梯度下降法
    * 直观理解BP （Backpropagation intuition）
    * 随机初始化
2. 习题测试
3. programming Assignments
    * Planar data classification with a hidden layer
4. 采访： lan Goodfellow

## Week4 - Deep Neural Networks
1. Deep Neural Network
    * Deep L-layer neural network
    * 深层网络中的前向传播
    * 核对矩阵的维数
    * Why deep representation
    * 搭建深层神经网络块
    * 前向和反向传播
    * 参数 vs 超参数
    * What does this have to do with the brain？
2. 习题练习
3. Programming Assignments
    * Building your Deep Neural Network: Step by Step
    * Deep Neural NetWork - Application


# 课程2 - Improving Deep Neural Networks Hyperparameter tuning Regularization and Optimization

## Week1 - 深度学习的应用层面（Practical aspects of Deep Learning）
1. Setting up your Machine Learning Application
    * 训练/开发/测试集
    * 偏差/方差
    * Basic Recipe for Machine Learning
2. Regularizing your neural network
    * Regularization
    * Why regularization reduces overfitting?
    * Dropout Regularization
    * Understanding Dropout
    * Other regularization
3. Setting up your optimization problem
    * 正则化输入
    * 梯度消失和梯度爆炸（Vanishing / Exploding gradients）
    * 权重初始化
    * 梯度的数值逼近
    * 梯度检查
    * Gradient Checking Implementation Notes
4. 习题练习
5. programming assignments
    1. 初始化（Initialization）
    2. 正则化 （Regularization）
    3. 梯度检查（Gradient Checking）

## Week2 - 最优化算法（Optimization algorithms）
1. 最优化算法（Optimization algorithms）
    * mini-batch 梯度下降 及理解
    * 指数加权平均 及理解（Exponentially weighted averages）
    * 指数加权平均的偏差修正（Bias correction in exponentially weighted averages）
    * 动量梯度下降(Gradient descent with momentum)
    * RMSprop （Root Mean Square prop）
    * Adam Optimization
    * Learning rate decay
    * The problem of local optima
2. 习题练习
3. programming assignments
    1. 最优化（Optimization）
4. 采访： Yuanqing Lin

## Week3 - Hyperparameter tuning, Batch Normalization and Programming Frameworks
1. Hyperparameter tuning
    * Tuning process
    * Using a appropriate scale to pick hyperparameters
    * Hyperparameters tuning in practice : Pandas vs. Caviar
2. Batch Normalization
    * Normalizing activations in a network
    * Fitting Batch Norm into a neural network
    * Why does Batch Norm work? 
    * Batch Norm at test time
3. Multi-class classification
    * Softmax Regression
    * Training a softmax classifier
4. Introduction to programming frameworks
    * Deep learning frameworks
    * TensorFlow
5. 习题练习
6. Programming assignment
    * Tensorflow


# 课程3 - Structuring Machine Learning Projects

## Week1 - ML Strategy (1)
1. Introduction to ML Strategy
    * Why ML Strategy
    * Orthogonalization
2. Setting up your goal
    * Single number evaluation metric
    * Satisficing and Optimizing metric
    * Train/dev/test distributions
    * Size of the dev and test sets
    * When to change dev/test sets and metrics
3. Comparing to human-level performance
    * Why human-level performance?
    * Avoidable bias （贝叶斯误差 - train set error）
    * Understanding human-level performance
    * Surpassing human-level performance
    * Improving your model performance
4. 机器学习飞行模拟器：Bird recognition in the city of Peacetopia (case study)
5. 采访： Andrej Karpathy interview

## Week2 - ML Strategy (2)
1. Error Analysis
    * Carrying out error analysis
    * Cleaning up incorrectly labeled data
    * Build your first system quickly, then iterate
2. Mismatched training and dev/test set
    * Training and testing on different distributions
    * Bias and Variance with mismatched data distributions
    * Addressing data mismatch
3. Learning from multiple tasks
    * Transfer learning (迁移误差)
    * Multi-task learning （多任务误差）
4. End-to-end deep learning
    * What is end-to-end deep learning?
    * Whether to use end-to-end deep learning
5. 机器学习飞行模拟器：Autonomous driving (case study)
6. 采访： Ruslan Salakhutdinov