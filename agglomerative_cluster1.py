import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory


# learn what a connectivity matrix is?
# create sample data
x , _ = make_blobs(n_samples= 300, centers= 3, random_state = 0)

# print(x)
print()
# print(_)
print()

# precomputed connectivity matrix
connectivity = kneighbors_graph(x, n_neighbors= 10, include_self= False)

print(connectivity)

# create a location to store cached files
memory = Memory("cache_dir_agg_cluster", verbose= 0)


# create an instance of agglomerative clustering
agc = AgglomerativeClustering(n_clusters = 3,metric="euclidean", 
memory = memory,connectivity = connectivity, compute_full_tree = True, linkage ="ward", distance_threshold= None,
compute_distances=True)

print(agc)

# fit the data to the model
agc.fit(x)

# cluster labels for each point
labels = agc.labels_

# plotting the data colored by clusters
plt.scatter(x[:, 0], x[:,  1], c= labels)

plt.show()


