from hdfs import InsecureClient

# Connect to HDFS
client = InsecureClient('http://localhost:9870', user='root')

# Create a directory on HDFS
client.makedirs('/user/input/')
print("Directory successfully created!")
