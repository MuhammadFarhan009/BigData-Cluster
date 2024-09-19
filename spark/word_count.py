from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("HDFS Word Count") \
    .getOrCreate()

# Read the file from HDFS
lines = spark.read.text("hdfs://hadoop-namenode:8020/input/test.txt").rdd.map(lambda r: r[0])

# Perform word count
word_counts = lines.flatMap(lambda line: line.split(" ")) \
                   .map(lambda word: (word, 1)) \
                   .reduceByKey(lambda a, b: a + b)

# Collect and print the results
for word, count in word_counts.collect():
    print(f"{word}: {count}")

# Stop the Spark session
spark.stop()