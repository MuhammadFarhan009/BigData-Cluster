from hdfs import InsecureClient
from hdfs.util import HdfsError
import logging
logging.basicConfig(level=logging.DEBUG)

try:
    # Connect to HDFS
    client = InsecureClient('http://localhost:9870', user='root')

    # Upload a file to HDFS
    client.upload('/user/input/', './hdfs/input/data.txt')
    print("File uploaded successfully!")
except HdfsError as e:
    print(f"An error occurred: {e}")
