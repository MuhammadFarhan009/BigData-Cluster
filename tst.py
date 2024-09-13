from hdfs import InsecureClient

# Connect to HDFS
client = InsecureClient('http://localhost:9870', user='hadoop')

# Create a new directory
client.makedirs('/input')
