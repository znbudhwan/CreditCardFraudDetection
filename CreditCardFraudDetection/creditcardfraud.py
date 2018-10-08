''' This a program that uses unsupervised machine learning in 
order to predict fraudulent credit card transactions using 
isolation forest algorithm and local outlier factor algorithm to 
detect anomalies '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest 
from sklearn.neighbors import LocalOutlierFactor 

data = pd.read_csv('creditcard.csv')
data = data.sample(frac = 0.35, random_state = 1)

# We can determine number of valid/fraudulent cases

# Dataset Class = 0/Valid or 1/Fraudulent

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

# Format dataset by filtering out the class for our unsupervised ml

columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]

# Target variable we are going to predict is the class variable

target = "Class"

creditCardTransactions = data[columns]
transactionValidity = data[target]

# Outlier detection methods

n_outliers = len(Fraud)
n_valid = len(Valid)

outlier_fraction = n_outliers / float(n_valid)

classifers = {
	"Isolation Forest": IsolationForest(
		max_samples = len(creditCardTransactions),
		contamination = outlier_fraction,
		random_state = 1),
	"Local Outlier Factor": LocalOutlierFactor(
		n_neighbors = 20,
		contamination = outlier_fraction)
}

# Create a model to fit the data

for i, (classifer_name, classifier) in enumerate(classifers.items()):

	if classifer_name == "Isolation Forest":
		classifier.fit(creditCardTransactions)
		scores_pred = classifier.decision_function(creditCardTransactions)
		validity_pred = classifier.predict(creditCardTransactions)
	else:
		validity_pred = classifier.fit_predict(creditCardTransactions)
		scores_pred = classifier.negative_outlier_factor_

	# Changing the values back to the original class format of 0/1
	validity_pred[validity_pred == 1] = 0
	validity_pred[validity_pred == -1] = 1

	# Compare the predicted results vs actual results
	n_errors = (validity_pred != transactionValidity).sum()

	print('{}: {}'.format(classifer_name, n_errors))
	print(accuracy_score(transactionValidity, validity_pred))
	print(classification_report(transactionValidity, validity_pred))
