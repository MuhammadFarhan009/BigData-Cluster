from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Text Classification with PySpark and HDFS") \
    .getOrCreate()

# Define HDFS paths
data_folder = "hdfs://hadoop-namenode:8020/user/root/input"
stopwords_path = "/user/root/input/stopwords.txt"

# Create a regex to extract the folder name from the file path
folder_regex = ".*/input/([^/]+)/.*"

# Read all CSV files from subfolders
df = spark.read.csv(f"{data_folder}/*/*.csv", header=True, inferSchema=True)

# Add a column to display the file path (for debugging purposes)
df = df.withColumn("file_path", input_file_name())

# Add a new column "Label" based on the folder name extracted from the file path
df = df.withColumn("Label", regexp_extract(input_file_name(), folder_regex, 1))

# Remove duplicates based on the 'Title' column (if it exists)
if "Title" in df.columns:
    df = df.dropDuplicates(["Title"])

# Ensure required columns exist
if "Content" not in df.columns or "Label" not in df.columns:
    print("Required columns ('Content', 'Label') are missing!")
    spark.stop()
    
if df.filter(col("Label") == "").count() > 0:
    print("Some labels are missing!")


# Read the stopwords file from HDFS
stopwords_rdd = spark.sparkContext.textFile(f"hdfs://hadoop-namenode:8020{stopwords_path}")
custom_stopwords = stopwords_rdd.collect()

# Preprocessing pipeline
# Tokenize, remove stopwords, calculate TF-IDF, and label index
tokenizer = Tokenizer(inputCol="Content", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered", stopWords=custom_stopwords)
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
labelIndexer = StringIndexer(inputCol="Label", outputCol="label")

# Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label", probabilityCol="probability", rawPredictionCol="rawPrediction")

# Build the pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, labelIndexer, lr])

# Split data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_df)

# Make predictions
predictions = model.transform(test_df)

# Select and display the required columns
predictions.select("Content", "Label", "prediction").show(10, truncate=False)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)


# Define the path to save the model
model_save_path = "hdfs://hadoop-namenode:8020/user/root/output/text_classification_model2"

# Save the trained model
model.write().overwrite().save(model_save_path)

print(f"Model saved to: {model_save_path}")
print(f"Accuracy: {accuracy}")
