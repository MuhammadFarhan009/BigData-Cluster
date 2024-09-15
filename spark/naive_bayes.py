from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("TextClassification").getOrCreate()

# Load and prepare the dataset
data = spark.read.csv("hdfs://hadoop-namenode:8020/hdfs/input/text_data.csv", header=True)

# Tokenize the text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_data = tokenizer.transform(data)

# TF-IDF feature extraction
hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)
tf_data = hashing_tf.transform(words_data)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(tf_data)
tfidf_data = idf_model.transform(tf_data)

# Prepare the data for classification
train_data, test_data = tfidf_data.randomSplit([0.8, 0.2])

# Train Logistic Regression model
lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(train_data)

# Test the model
predictions = lr_model.transform(test_data)
predictions.select("text", "label", "prediction").show()
