# CreditCardFraudDetection
Uses sklearn machine learning algorithms, isolation forest and local outlier factors, to detect fraudulent transactions in a Kaggle dataset

## Logic Involved

There was a lot of research and learning that was required for this
project to be a success. I had to figure out what types of methods I can use to find fraudulent data, and moreso, be able to have a machine learning model do it.

We will be comparing two methods of finding anomalous data, which will be our fraudulent credit card transactions, through the use of isolation forests and local outlier factors.

### Isolation Forests

Isolation Forest Algorithm is an unsupervised algorithm that uses decision trees, which creates partitions by randomly selecting a feature and then a random split value between the minimum and maximum of the selected feature. In our case IF would create random partitons based off either the time or amount of a transaction.

Outliers in our dataset, which are our fraudulent values, are less frequent by a large margin compared to our valid values, and are noticeably different in the select features of time and amount. Using the IF deciscion tree, paths to fraudulent values are noticeably short relative to normal values. The runtime of this algorithm is in linear time because we are only trying to isolate the anomalies. 

The idea of using random partitioning is that anomalies will require less partitions than the average normal observations.

### Local Outlier Factor

In conjunction with Isolation Forest, we are going to be using the local outlier factor algorithm which finds anomalous data by measuring the deviation of a given data point with respect to its neighbors. 

![alt text](http://url/to/img.png)
