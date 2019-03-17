# another source with genetics https://github.com/vanhooser/TF-Genetic
# source wtih genetics https://github.com/amirdeljouyi/Genetic-Algorithm-on-K-Means-Clustering
# source of this file code  https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/KMeansClustering

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

num_points = 100
dimensions = 2
points = np.random.uniform(0, 1000, [num_points, dimensions])

def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

num_clusters = 2
kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_clusters, use_mini_batch=False)

# train
num_iterations = 10
previous_centers = None
for _ in xrange(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print 'delta:', cluster_centers - previous_centers
  previous_centers = cluster_centers
#   print 'score:', kmeans.score(input_fn)
print 'cluster centers:', cluster_centers

# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(points):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  color = ("red", "green")[cluster_index == 0]
  plt.plot(point[0], point[1], marker='x', markersize=15, color=color)
#   print 'point:', point, 'is in cluster', cluster_index, 'centered at', center



# Label the graph axes.
plt.ylabel("y")
plt.xlabel("x")

# Plot a scatter plot from our data sample.
firstEls = [];
secondEls = [];

for i in range(0, num_points):
    firstEls.append(points[i][0])

for i in range(0, num_points):
    secondEls.append(points[i][1])

plt.scatter(firstEls, secondEls)

plt.plot(cluster_centers[0][0], cluster_centers[0][1], marker='o', markersize=15, color="red")
plt.plot(cluster_centers[1][0], cluster_centers[1][1], marker='o', markersize=15, color="green")

# Display graph.
plt.show()

# to run script run ipython classification.py

