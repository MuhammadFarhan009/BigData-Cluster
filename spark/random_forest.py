from pyspark.ml.classification import RandomForestClassifier

# Train a Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20)
rf_model = rf.fit(train_data)

# Test the model
rf_predictions = rf_model.transform(test_data)
rf_predictions.select("text", "label", "prediction").show()
