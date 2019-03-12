# another source with genetics https://github.com/vanhooser/TF-Genetic
# source wtih genetics https://github.com/amirdeljouyi/Genetic-Algorithm-on-K-Means-Clustering
# source of this file code  https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/KMeansClustering

import numpy as np
import tensorflow as tf

num_points = 100
dimensions = 8
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
  print 'score:', kmeans.score(input_fn)
print 'cluster centers:', cluster_centers

# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(points):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  print 'point:', point, 'is in cluster', cluster_index, 'centered at', center



# examples of tensorboard from https://itnext.io/how-to-use-tensorboard-5d82f8654496

sess = tf.Session()


x_matrix = tf.get_variable('x_matrix', shape=[30, 40], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

histogram_summary = tf.summary.histogram('My_first_histo_summary', x_matrix)
init = tf.global_variables_initializer()
writer = tf.summary.FileWriter('./points', sess.graph)

for step in range(100):
    sess.run(init)
    summary2 = sess.run(histogram_summary)
    writer.add_summary(summary2, step)


# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

# test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
# tf.global_variables_initializer().run()

