from pyspark.sql.functions import monotonically_increasing_id
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
    ("Melalui pemanfaatan teknologi canggih seperti artificial intelligence (AI/kecerdasan buatan) dan machine learning (ML/pembelajaran mesin), keduanya ingin mengoptimalkan investasi jaringan serta mengalokasikan sumber daya secara efisien di area prioritas, menghadirkan efisiensi operasional, dan memberikan pengalaman terbaik bagi pelanggan.",),
    ("Personel prajurit TNI AU dibawah komando Komandan Lanud Husein Sastranegara Kolonel Pnb Alfian itu bergerak memberikan bantuan logistik untuk membantu masyarakat yang terdampak musibah bencana longsor akibat cuaca ekstrim yang melanda sejumlah wilayah di Jawa Barat beberapa waktu lalu. Bantuan ini merupakan wujud kepedulian dan respons cepat terhadap kebutuhan para korban, terutama yang berada di wilayah sulit dijangkau pasca bencana.",),
    ("Dia menambahkan program-program Kowani sebelumnya seperti Gerakan Ibu Bangsa Percepatan Penurunan Stunting, Gerakan Ibu Bangsa berwakaf, Gerakan Ibu Bangsa anti tembakau/zat adiktif, Gerakan Ibu Bangsa untuk Jaminan Sosial Ketenagakerjaan, Gerakan Ibu Bangsa Pemberdayaan UMKM Perempuan melalui Kowani Fair, Gerakan Ibu Bangsa Menolak LGBT di Indonesia, Gerakan Ibu Bangsa Mendorong Kepemimpinan Perempuan dan lainnya, dapat terus dilanjutkan sebagai bagian dari upaya mencapai Indonesia Emas 2045",),
    ("Dukung Pariwisata Indonesia, Waketum Koordinator Kadin Sebut Stakeholders Harus Bekerja Sama",)
]

# Define schema for the test DataFrame
test_columns = ["Content"]

# Create DataFrame from test data
test_df = spark.createDataFrame(test_data, test_columns)

# Add sequential numbering
test_df = test_df.withColumn("row_number", monotonically_increasing_id() + 1)

# Rename columns to replace 'Content' with sequential numbers
test_df = test_df.withColumn("Content", test_df["row_number"]).drop("row_number")

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
