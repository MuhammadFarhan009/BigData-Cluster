from pyspark.ml.feature import HashingTF, IDF

# Load data into a DataFrame
data = spark.read.text("hdfs://hadoop-namenode:8020/hdfs/path/data.txt").rdd

# Create a TF-IDF vectorizer
hashing_tf = HashingTF(inputCol="text", outputCol="raw_features", numFeatures=20)
tf = hashing_tf.transform(data)

# Fit an IDF model to the data
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(tf)
tfidf_data = idf_model.transform(tf)
