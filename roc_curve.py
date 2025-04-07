# -------------------------------------------------------------------------
# AUTHOR: Vi Nguyen
# FILENAME: roc_curve 
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 120 minutes
# -----------------------------------------------------------*/
#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to
# complete this code.
#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here
# data_training = ?
# Read the dataset cheat_data.csv and prepare the data_training numpy array
data = pd.read_csv('cheat_data.csv')
data['Taxable Income'] = data['Taxable Income'].str.replace('k', '', regex=False).astype(float)
data['Refund'] = data['Refund'].map({'Yes': 1, 'No': 0})
data['Cheat'] = data['Cheat'].map({'Yes': 1, 'No': 0})

# One-hot encode 'Marital Status'
marital_encoded = pd.get_dummies(data['Marital Status'], prefix='Marital')


# transform the original training features to numbers and add them to the 5D array
# X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The
# feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# Combine all features
X = pd.concat([data['Refund'], marital_encoded, data['Taxable Income']], axis=1).values
y = data['Cheat'].values

# --> add your Python code here
# X = ?
# transform the original training classes to numbers and add them to the vector y.
# For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
# --> add your Python code here
# y = ?
# split into train/test sets using 30% for test
# --> add your Python code here
# trainX, testX, trainy, testy = train_test_split(X, y, test_size = ?)
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)

# generate random thresholds for a no-skill prediction (random classifier)
# --> add your Python code here
# ns_probs = ?
# fit a decision tree model by using entropy with max depth = 2
# clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=?)
# clf = clf.fit(trainX, trainy)

ns_probs = [0 for _ in range(len(testy))]

# Train Decision Tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
# dt_probs = clf.predict_proba(testX)
dt_probs = clf.predict_proba(testX)[:, 1]  # probability for class 1
# keep probabilities for the positive outcome only
# --> add your Python code here
# dt_probs = ?
# calculate scores by using both classifiers (no skilled and decision tree)
# ns_auc = roc_auc_score(testy, ns_probs)
# dt_auc = roc_auc_score(testy, dt_probs)
# AUC Scores
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)
print('No Skill AUC: %.3f' % ns_auc)
print('Decision Tree AUC: %.3f' % dt_auc)

# ROC Curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# Plot
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()