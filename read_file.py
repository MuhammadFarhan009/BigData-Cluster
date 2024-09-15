from hdfs import InsecureClient

# Connect to HDFS Namenode
client = InsecureClient('http://localhost:9870', user='root')

# Reading a file from HDFS
with client.read('/hdfs/path/file.txt') as reader:
    data = reader.read()
    print(data)
