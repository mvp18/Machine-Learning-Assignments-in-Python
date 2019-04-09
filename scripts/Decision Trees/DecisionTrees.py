# Execute as 'python3 DecisionTrees.py'
# Uncomment last line of code to see structure of decision tree in preorder format                       

import numpy as np
import math

def calc_data_entropy(data):
	labels = data[:,-1]
	positive_examples=np.count_nonzero(labels)
	negative_examples=len(labels)-positive_examples
	if positive_examples==0 or negative_examples==0:
		return 0
	else:
		pos_proportion = positive_examples/float(len(labels))
		neg_proportion = negative_examples/float(len(labels))
		return -(pos_proportion*math.log2(pos_proportion)+neg_proportion*math.log2(neg_proportion))

def info_gain(data,x,data_entropy):
	data_subset_zero_x = data[data[:,x]==0]
	data_subset_one_x = data[data[:,x]==1]
	subset_entropy_zero_x = calc_data_entropy(data_subset_zero_x)
	subset_entropy_one_x = calc_data_entropy(data_subset_one_x)
	weight_of_subset_zero_x = data_subset_zero_x.shape[0]/float(data.shape[0])
	weight_of_subset_one_x = data_subset_one_x.shape[0]/float(data.shape[0])
	weighted_sum_of_entropies = weight_of_subset_zero_x*subset_entropy_zero_x+weight_of_subset_one_x*subset_entropy_one_x
	info_gain_x = data_entropy - weighted_sum_of_entropies
	return info_gain_x

def print_tree(tree): # Preorder traversal
	    if tree == None: 
	    	return
	    print(tree.val)
	    print_tree(tree.left_child)
	    print_tree(tree.right_child)

def find_best_classifier_attribute(data,data_entropy):
	attribute_entropies = []
	for i in range(data.shape[1]-1):
		attribute_entropies.append(info_gain(data,i,data_entropy))
	return np.argmax(attribute_entropies) # returns index of first occurence of maximum

class node:
	def __init__(self,value = None):
		self.val = value
		self.left_child = None
		self.right_child = None

def expand_tree(data, decision_tree):
	if calc_data_entropy(data) == 0:
		if np.count_nonzero(data[:,-1])==0:
			if decision_tree == None:
				decision_tree = node('No')
			else:   
				decision_tree.val = 'No'
		else:
			if decision_tree == None:
				decision_tree = node('Yes')
			else: 
				decision_tree.val = 'Yes'       
	else:
		best_classifier_attribute = find_best_classifier_attribute(data,calc_data_entropy(data))
		attribute_info_gain = info_gain(data,best_classifier_attribute,calc_data_entropy(data))
		if attribute_info_gain>0:
			if decision_tree == None:
				decision_tree = node(best_classifier_attribute)
			else:
				decision_tree.val = best_classifier_attribute
			decision_tree.left_child = node()
			expand_tree(data[data[:,best_classifier_attribute]==0], decision_tree.left_child)
			decision_tree.right_child = node()
			expand_tree(data[data[:,best_classifier_attribute]==1], decision_tree.right_child)
		elif attribute_info_gain==0: # When there is only value of a particular attribute
			if decision_tree == None:
				decision_tree = node(best_classifier_attribute)
			else:
				decision_tree.val = best_classifier_attribute
			only_attribute = data[0,best_classifier_attribute]
			most_dominant_label = np.bincount(data[:,-1]).argmax()
			if only_attribute==0:
				decision_tree.right_child = node()
				if most_dominant_label==0:
					decision_tree.right_child.val = 'No'
				elif most_dominant_label==1:
					decision_tree.right_child.val = 'Yes'
				decision_tree.left_child = node()
				expand_tree(data[data[:,best_classifier_attribute]==0], decision_tree.left_child)
			elif only_attribute==1:
				decision_tree.left_child = node()
				if most_dominant_label==0:
					decision_tree.left_child.val = 'No'
				elif most_dominant_label==1:
					decision_tree.left_child.val = 'Yes'
				decision_tree.right_child = node()
				expand_tree(data[data[:,best_classifier_attribute]==1], decision_tree.right_child)
					
def write_predictions(test_instance, decision_tree, file):
	if decision_tree.val == 'Yes': # Leaf Node
		file.write('1 ')
		return
	if decision_tree.val == 'No': # Leaf Node
		file.write('0 ')
		return
	if test_instance[decision_tree.val]==0:
		write_predictions(test_instance, decision_tree.left_child, file)
	elif test_instance[decision_tree.val]==1:
		write_predictions(test_instance, decision_tree.right_child, file)

def generate_outfile(test_data, decision_tree):
	outfile = open('DecisionTrees.out', 'w')
	for i in range(test_data.shape[0]):
		write_predictions(test_data[i], decision_tree, outfile)
	outfile.close()    

train_data = np.genfromtxt('./data2.csv', delimiter = ',')

test_data = np.genfromtxt('./test2.csv', delimiter = ',')

decision_tree = node()

expand_tree(train_data, decision_tree)

generate_outfile(test_data, decision_tree)

#print_tree(decision_tree)