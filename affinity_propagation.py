import time
import sklearn

from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# create sample data
x, _ = make_blobs(n_samples= 3000, centers = 3, random_state= 0)

# print(x)
print(_)

# create an instance of affinity propagation
ap = AffinityPropagation(random_state=42)

# fiting the data -> this also performs clustering
ap.fit(x)

# predict the cluster labels for the same data (optional, just to demonstrate)
labels = ap.predict(x)

print(labels)

# plotting the data colored by clusters
plt.scatter(x[:, 0], x[:, 1] , c= labels)
plt.scatter(ap.cluster_centers_[:, 0], ap.cluster_centers_[:, 1] , s= 300, c ="red")
plt.show()