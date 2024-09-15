from hdfs import InsecureClient

# Connect to HDFS Namenode
client = InsecureClient('http://localhost:9870', user='root')

# Upload a file to HDFS
client.upload('/hdfs/path/file.txt', '/local/path/file.txt')
print("File uploaded successfully!")
