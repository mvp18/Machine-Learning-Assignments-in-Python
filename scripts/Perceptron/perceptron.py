# Execute as 'python3 perceptron.py'

import numpy as np

# Data Loading
dataset = np.genfromtxt('./data6.csv',delimiter=',')
test_data = np.genfromtxt('./test6.csv',delimiter=',')

num_attributes = dataset.shape[1]-1
X_train = [dataset[i][:num_attributes] for i in range(len(dataset))]
y_train = [int(dataset[i][-1]) for i in range(len(dataset))]
X_test = [test_data[i][:num_attributes] for i in range(len(test_data))]

class Perceptron(): 
      
    def __init__(self, random_state = 1): 
        
        self.random_state = random_state
  
    def sigmoid(self, x): 
        return 1/(1+np.exp(-x)) 
   
    def sigmoid_derivative(self, x): 
        
        derivative = self.sigmoid(x)*(1-self.sigmoid(x))
        
        return derivative
  
    def forward_propagation(self, inputs): 
        
        return self.sigmoid(np.dot(inputs, self.weights[1:])+self.weights[0]) 
      
    # training the neural network. 
    def train(self, X_train, y_train, learning_rate, num_train_iterations): 
                                  
        random_generator = np.random.RandomState(self.random_state)
        
        x_columns = num_attributes+1
        
        self.weights = random_generator.normal(loc=0.0, scale=0.001, size=x_columns)
        
        # print('Weights before training : ', self.weights)
        
        for iteration in range(num_train_iterations):
            
            for train_inputs,y_actual in zip(X_train, y_train):
            
                y_predicted = self.forward_propagation(train_inputs) 

                delta = learning_rate*(y_actual - y_predicted)*self.sigmoid_derivative(y_predicted) 

                # Adjust the weight matrix 
                self.weights[1:] += delta*train_inputs

                self.weights[0] += delta

perceptron = Perceptron()

# Learning rate and number of epochs can be altered for different results

perceptron.train(X_train, y_train, 0.1, 10)

# print('Weights after training : ', perceptron.weights)

outfile = open('perceptron.out','w')
for X in X_test:
    test_label = np.round(perceptron.forward_propagation(X))
    # print(test_label)
    if test_label == 1:
        outfile.write('1 ')
    else:
        outfile.write('0 ')
outfile.close()