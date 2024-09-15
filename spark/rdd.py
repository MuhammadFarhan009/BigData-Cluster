# Load data from HDFS into an RDD
rdd = sc.textFile("hdfs://hadoop-namenode:8020/hdfs/path/file.txt")

# Perform a transformation
filtered_rdd = rdd.filter(lambda line: "keyword" in line)

# Perform an action
result = filtered_rdd.collect()
print(result)
