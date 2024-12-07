from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Test Text Classification Model") \
    .getOrCreate()

# Define the path to the saved model
model_load_path = "hdfs://hadoop-namenode:8020/user/root/output/text_classification_model"

# Load the saved model
model = PipelineModel.load(model_load_path)

# Access the StringIndexer stage to retrieve labels
label_indexer = model.stages[-2]  # Assuming StringIndexer is the second-last stage
labels = label_indexer.labels     # This is the mapping of index to label

# Create test data inline
test_data = [
    ("WhatsApp menjelaskan ikon typing bubble itu berlaku baik untuk obrolan individu maupun di dalam grup. Saat lawan bicara mengetik, WhatsApp juga akan menampilkan foto profil di samping typing bubble tersebut.",),
    
]

# Define schema for the test DataFrame
test_columns = ["Content"]

# Create DataFrame from test data
test_df = spark.createDataFrame(test_data, test_columns)

# Make predictions
predictions = model.transform(test_df)

# Map numeric predictions to their label representations
def map_prediction_to_label(prediction):
    return labels[int(prediction)]

# Register the mapping function as a UDF
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
map_to_label_udf = udf(map_prediction_to_label, StringType())

# Add a column with label representations
predictions_with_labels = predictions.withColumn(
    "predicted_label", map_to_label_udf(predictions["prediction"])
)

# Select and display results
predictions_with_labels.select("Content", "prediction", "predicted_label").show(10, truncate=False)

# Stop SparkSession
spark.stop()
