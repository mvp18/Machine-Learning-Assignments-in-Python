# Execute as 'python3 NaiveBayes.py'

import numpy as np

def calc_attr_prob_given_y(vmap,Class):
    attr_prob = 1.0
    if Class==0:
        for index in range(len(vmap)):
            if vmap[index] == 1: #feature present
                attr_prob *= X_present_prob_class_0[index] 
            elif vmap[index] == 0: #feature absent
                attr_prob *= X_absent_prob_class_0[index]
    elif Class==1:
        for index in range(len(vmap)):
            if vmap[index] == 1: #feature present
                attr_prob *= X_present_prob_class_1[index] 
            elif vmap[index] == 0: #feature absent
                attr_prob *= X_absent_prob_class_1[index]
    return attr_prob

def calculate_class_posteriors(feature_vector):
    attr_prob_class_0 = calc_attr_prob_given_y(feature_vector, Class=0)
    attr_prob_class_1 = calc_attr_prob_given_y(feature_vector, Class=1)
    
    posterior_prob_class_0 = attr_prob_class_0*prior_probability_class_0
    posterior_prob_class_1 = attr_prob_class_1*prior_probability_class_1
    
    return posterior_prob_class_0, posterior_prob_class_1

def classify_test_vector(test_data):
    outfile = open('NaiveBayes.out','w')
    for test_vector in test_data:
        posterior_prob_class_0, posterior_prob_class_1 = calculate_class_posteriors(test_vector)
        if posterior_prob_class_0 > posterior_prob_class_1:
            outfile.write('0 ')
        else:
            outfile.write('1 ')
    outfile.close()

# Data Loading
dataset = np.genfromtxt('./data3.csv',delimiter=',')
test_data = np.genfromtxt('./test3.csv',delimiter=',')
# Data Separation
num_attributes = dataset.shape[0]-1
X_train = [dataset[i][:num_attributes] for i in range(len(dataset))]
y_train = [int(dataset[i][-1]) for i in range(len(dataset))]
X_test = [test_data[i][:num_attributes] for i in range(len(test_data))]
# Separate train data into classes
X_class_0 = [X_train[i] for i in range(len(X_train)) if y_train[i]==0]
X_class_1 = [X_train[i] for i in range(len(X_train)) if y_train[i]==1]
# Conditional Probabilities with Laplacian smoothing
X_present_prob_class_0 = (np.sum(X_class_0,axis=0)+1)/(2+float(len(X_class_0)))
X_present_prob_class_1 = (np.sum(X_class_1, axis=0)+1)/(2+float(len(X_class_1)))
X_absent_prob_class_0 = (len(X_class_0)-np.sum(X_class_0,axis=0)+1)/(2+float(len(X_class_0)))
X_absent_prob_class_1 = (len(X_class_1)-np.sum(X_class_1,axis=0)+1)/(2+float(len(X_class_1)))
# Prior Probabilties
num_class_0 = float(len(X_class_0))
num_class_1 = float(len(X_class_1))
prior_probability_class_0 = num_class_0 / (num_class_0 + num_class_1)
prior_probability_class_1 = num_class_1 / (num_class_0 + num_class_1)
# Execute
classify_test_vector(X_test)

