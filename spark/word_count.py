from pyspark import SparkConf, SparkContext

# Create Spark context
conf = SparkConf().setAppName("WordCount").setMaster("spark://localhost:7077")
sc = SparkContext(conf=conf)

# Load data from HDFS
data = sc.textFile("hdfs://hadoop-namenode:8020/hdfs/path/file.txt")

# Count words
words = data.flatMap(lambda line: line.split(" "))
word_counts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

# Save result to HDFS
word_counts.saveAsTextFile("hdfs://hadoop-namenode:8020/hdfs/path/word_counts")
print("Word count completed!")
