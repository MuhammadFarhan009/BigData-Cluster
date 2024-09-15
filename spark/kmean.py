from pyspark.ml.clustering import KMeans

# Train a K-Means model
kmeans = KMeans(k=3, featuresCol="features")
model = kmeans.fit(train_data)

# Show the cluster centers
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Predict cluster for test data
predictions = model.transform(test_data)
predictions.select("text", "prediction").show()
