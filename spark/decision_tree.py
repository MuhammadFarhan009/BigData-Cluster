from pyspark.ml.classification import DecisionTreeClassifier

# Train a Decision Tree model
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_model = dt.fit(train_data)

# Test the model
dt_predictions = dt_model.transform(test_data)
dt_predictions.select("text", "label", "prediction").show()
