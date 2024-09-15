from pyspark.ml.clustering import PowerIterationClustering

# Train a Power Iteration Clustering model
pic = PowerIterationClustering(k=3, maxIter=20)
model = pic.assignClusters(train_data)

# Show the clusters
model.show()
