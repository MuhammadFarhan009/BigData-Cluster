from pyspark.ml.classification import LinearSVC

# Train a Linear SVM model
svm = LinearSVC(featuresCol="features", labelCol="label")
svm_model = svm.fit(train_data)

# Test the model
svm_predictions = svm_model.transform(test_data)
svm_predictions.select("text", "label", "prediction").show()
