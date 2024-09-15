# Reading a file from HDFS
with client.read('/hdfs/path/file.txt') as reader:
    data = reader.read()
    print(data)
