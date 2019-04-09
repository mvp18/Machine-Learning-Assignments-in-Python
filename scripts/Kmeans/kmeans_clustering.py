# Execute as 'python3 kmeans_clustering.py'

import numpy as np

dataset = np.genfromtxt('./data7.csv',delimiter=',')

class KMeans(): 
	  
	def __init__(self, num_clusters = 2): 
		
		self.k = num_clusters
		
		  
	def calc_euclidean_distance(self, x, y): 
		
		sum_of_squares = 0
		
		for i in range(x.shape[0]):
			sum_of_squares += (x[i]-y[i])**2
			
		euclidean_distance = sum_of_squares**0.5
		
		return euclidean_distance

	def find_closest_cluster(self, x, cluster1_centroid, cluster2_centroid): 
		
		dist_first_cluster = self.calc_euclidean_distance(x, cluster1_centroid)
		
		dist_second_cluster = self.calc_euclidean_distance(x, cluster2_centroid)
		
		if dist_first_cluster<dist_second_cluster:
			return 1
		else:
			return 2
  
	def update_cluster_centroid(self, cluster): 
		
		cluster_centroid = np.mean(cluster, axis = 0)
			
		return cluster_centroid
   
	def assign_data_points_to_cluster(self, cluster1_centroid, cluster2_centroid): 
								  
		cluster1 = []
		cluster1_index = []
		cluster2 = []
		cluster2_index = []
		
		for i in range(dataset.shape[0]):
			
			nearest_cluster = self.find_closest_cluster(dataset[i], cluster1_centroid, cluster2_centroid)
			
			if nearest_cluster == 1:
				cluster1.append(dataset[i])
				cluster1_index.append(i)
			elif nearest_cluster == 2:
				cluster2.append(dataset[i])
				cluster2_index.append(i)
				
		cluster1 = np.asarray(cluster1)
		
		cluster2 = np.asarray(cluster2)

		cluster_1_centroid = self.update_cluster_centroid(np.asarray(cluster1))
		
		cluster_2_centroid = self.update_cluster_centroid(np.asarray(cluster2))
				
		return cluster1_index, cluster_1_centroid, cluster2_index, cluster_2_centroid


# Output would depend on cluster centroid intialization and number of iterations

random_data_point_1 = np.random.randint(dataset.shape[0])
		
cluster1_centroid_init = dataset[random_data_point_1]

random_data_point_2 = np.random.randint(dataset.shape[0])

while(random_data_point_2 == random_data_point_1):

	random_data_point_2 = np.random.randint(dataset.shape[0])

cluster2_centroid_init = dataset[random_data_point_2]

kmeans = KMeans()

num_iterations = 1000 # hyperparameter

for iteration in range(num_iterations//2):
	if iteration==0:
		cluster1_indices, cluster1_centroid, cluster2_indices, cluster2_centroid = kmeans.assign_data_points_to_cluster(cluster1_centroid_init, cluster2_centroid_init)
		old_centroid1 = cluster1_centroid
		old_centroid2 = cluster2_centroid
	else:
		# print(old_centroid2, old_centroid1)
		old_cluster1_indices, old_cluster1_centroid, old_cluster2_indices, old_cluster2_centroid = kmeans.assign_data_points_to_cluster(old_centroid1, old_centroid2)
		if kmeans.calc_euclidean_distance(old_centroid1, old_cluster1_centroid)<=0.005 and kmeans.calc_euclidean_distance(old_centroid2, old_cluster2_centroid)<=0.005:
			new_cluster1_indices = old_cluster1_indices
			new_cluster2_indices = old_cluster2_indices
			break

		new_cluster1_indices, new_cluster1_centroid, new_cluster2_indices, new_cluster2_centroid = kmeans.assign_data_points_to_cluster(old_cluster1_centroid, old_cluster2_centroid)
		old_centroid1 = new_cluster1_centroid
		old_centroid2 = new_cluster2_centroid
		# print(new_cluster1_centroid, new_cluster2_centroid)

		if kmeans.calc_euclidean_distance(new_cluster1_centroid, old_cluster1_centroid)<=0.005 and kmeans.calc_euclidean_distance(new_cluster2_centroid, old_cluster2_centroid)<=0.005:
			break
		# if new_cluster1_indices == old_cluster1_indices and new_cluster2_indices == old_cluster2_indices:
		# 	break

cluster1_final = new_cluster1_indices

cluster2_final = new_cluster2_indices

outfile = open('kmeans_clustering.out','w')
for i in range(dataset.shape[0]):
	if i in cluster1_final:
		outfile.write('1 ')
	elif i in cluster2_final:
		outfile.write('2 ')
outfile.close()
