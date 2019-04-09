from sklearn.cluster import KMeans
import numpy as np

dataset = np.genfromtxt('./data7.csv',delimiter=',')

kmeans = KMeans(n_clusters=2, random_state=0, max_iter=1000).fit(dataset)

print(kmeans.labels_)