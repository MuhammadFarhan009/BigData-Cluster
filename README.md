# Spark ML Project

This project demonstrates various machine learning models and data processing techniques using Apache Spark.

## Project Structure

```
docker-compose.yml
hdfs/
    input/
    output/
put_file.py
read_file.py
README.md
spark/
    decision_tree.py
    feature_extraction.py
    kmean.py
    lda.py
    naive_bayes.py
    pic.py
    random_forest.py
    rdd.py
    svm.py
    text_classification.py
    word_count.py
    word2vec.py
```

## Setup

1. **Install Dependencies**: Ensure you have Python and the required libraries installed. You can install the dependencies using pip:
    ```sh
    pip install -r requirements.txt
    ```

2. **Start HDFS and Spark**: Use `docker-compose` to start HDFS and Spark services.
    ```sh
    docker-compose up
    ```

## Scripts

### Data Handling

- **Upload a file to HDFS**: Use [`put_file.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fput_file.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/put_file.py") to upload a file to HDFS.
    ```sh
    python put_file.py
    ```

- **Read a file from HDFS**: Use [`read_file.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fread_file.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/read_file.py") to read a file from HDFS.
    ```sh
    python read_file.py
    ```

### Machine Learning Models

- **Decision Tree**: Train and test a Decision Tree model using [`spark/decision_tree.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Fdecision_tree.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/decision_tree.py").
    ```sh
    python spark/decision_tree.py
    ```

- **Support Vector Machine (SVM)**: Train and test an SVM model using [`spark/svm.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Fsvm.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/svm.py").
    ```sh
    python spark/svm.py
    ```

- **Word2Vec**: Train a Word2Vec model using [`spark/word2vec.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Fword2vec.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/word2vec.py").
    ```sh
    python spark/word2vec.py
    ```

- **Random Forest**: Train and test a Random Forest model using [`spark/random_forest.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Frandom_forest.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/random_forest.py").
    ```sh
    python spark/random_forest.py
    ```

- **Latent Dirichlet Allocation (LDA)**: Train an LDA model using [`spark/lda.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Flda.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/lda.py").
    ```sh
    python spark/lda.py
    ```

- **Naive Bayes**: Train and test a Naive Bayes model using [`spark/naive_bayes.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Fnaive_bayes.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/naive_bayes.py").
    ```sh
    python spark/naive_bayes.py
    ```

- **Text Classification**: Perform text classification using Logistic Regression in [`spark/text_classification.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Ftext_classification.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/text_classification.py").
    ```sh
    python spark/text_classification.py
    ```

### Data Processing

- **Feature Extraction**: Extract features using TF-IDF in [`spark/feature_extraction.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Ffeature_extraction.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/feature_extraction.py").
    ```sh
    python spark/feature_extraction.py
    ```

- **RDD Operations**: Perform RDD operations using [`spark/rdd.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Frdd.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/rdd.py").
    ```sh
    python spark/rdd.py
    ```

- **Word Count**: Count words in a text file using [`spark/word_count.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fspark%2Fword_count.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/spark/word_count.py").
    ```sh
    python spark/word_count.py
    ```

## HDFS Structure

- **Input Directory**: Place input files in [`hdfs/input/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fhdfs%2Finput%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/hdfs/input/").
- **Output Directory**: Output files will be saved in [`hdfs/output/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fworkspaces%2FBigData-Project%2Fhdfs%2Foutput%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22f9fbdf57-544e-432c-90cc-f6f3dc5a6e67%22%5D "/workspaces/BigData-Project/hdfs/output/").

