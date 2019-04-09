# Roll 16EE10056
# Name Soumava Paul
# Assignment 4
# Execute as 'python3 16EE10056_4.py'

import numpy as np

train_data = np.genfromtxt('./data4.csv',delimiter=',')
test_data = np.genfromtxt('./test4.csv',delimiter=',')

def calculate_euclidean_distance(input_array, test_array):
    sum_of_squares = 0
    for i in range(test_array.shape[0]):
        sum_of_squares += (input_array[i]-test_array[i])**2
    euclidean_distance = sum_of_squares**0.5
    return euclidean_distance

def find_5_nearest_neighbours(train_data, test_array):
    euclidean_distances = np.zeros(train_data.shape[0])
    for i in range(train_data.shape[0]):
        euclidean_distances[i] = calculate_euclidean_distance(train_data[i], test_array)
    five_nearest_neighbours = euclidean_distances.argsort()[:5]
    return five_nearest_neighbours

def find_dominant_class_and_classify(test_data, train_data):
    outfile = open('16EE10056_4.out','w')
    for k in range(test_data.shape[0]):
        nearest_neighbours = find_5_nearest_neighbours(train_data, test_data[k])
        classes = np.zeros(nearest_neighbours.shape[0])
        for i in range(nearest_neighbours.shape[0]):
            classes[i] = train_data[nearest_neighbours[i]][train_data.shape[1]-1]
        class_value = round(np.mean(classes))
        outfile.write(str(class_value)+' ')
    outfile.close()

find_dominant_class_and_classify(test_data, train_data)