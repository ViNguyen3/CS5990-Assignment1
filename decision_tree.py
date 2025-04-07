# -----------------------------------------------------------*/
# AUTHOR: Vi Nguyen
# FILENAME: decision_tree.py 
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: like 2 to 3 days 
# -----------------------------------------------------------*/
#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to
# complete this code.
#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']
test_file = "cheat_test.csv"


#read the test data and add this data to data_test NumPy
test_df = pd.read_csv(test_file)
test_df['Taxable Income'] = test_df['Taxable Income'].str.replace('k', '', regex=False).astype(float)
test_df['Refund'] = test_df['Refund'].map({'Yes': 1, 'No': 0})
test_df['Cheat'] = test_df['Cheat'].map({'Yes': 1, 'No': 2})
test_marital = pd.get_dummies(test_df['Marital Status'], prefix='Marital')
X_test = pd.concat([test_df['Refund'], test_marital, test_df['Taxable Income']], axis=1).values
y_test = test_df['Cheat'].values

# Store all final average accuracies
final_accuracies = []

for ds in dataSets:
# X = []
# Y = []
# df = pd.read_csv(ds, sep=',', header=0) 
#reading a dataset eliminating the header (Pandas library) data_training = np.array(df.values)[:,1:] 
#creating a training matrix without the id (NumPy library)
#transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
#Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
#be converted to a float.
# X =
#transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
    df = pd.read_csv(ds)
    df['Taxable Income'] = df['Taxable Income'].str.replace('k', '', regex=False).astype(float)
    df['Refund'] = df['Refund'].map({'Yes': 1, 'No': 0})
    df['Cheat'] = df['Cheat'].map({'Yes': 1, 'No': 2})
    marital = pd.get_dummies(df['Marital Status'], prefix='Marital')
    X = pd.concat([df['Refund'], marital, df['Taxable Income']], axis=1).values
    Y = df['Cheat'].values
    accuracies = []


#loop your training and test tasks 10 times here
    for i in range (10):
        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        if i == 0:
            plt.figure(figsize=(10, 6))
            #plotting the decision tree
            tree.plot_tree(clf, feature_names=['Refund', 'Marital_Divorced', 'Marital_Married', 'Marital_Single', 'Taxable Income'],
                            class_names=['Yes', 'No'], filled=True, rounded=True)
            plt.title(f"Decision Tree from {ds}")
            plt.show()

        # Evaluate on test data
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test,  predictions)
        accuracies.append(accuracy)

    avg_accuracy = np.mean(accuracies)
    final_accuracies.append((ds, avg_accuracy))


# Print final average accuracies
print("\nFinal classification performance (average accuracy over 10 runs):")
for file, acc in final_accuracies:
    print(f"{file}: {acc:.3f}")

    

#read the test data and add this data to data_test NumPy
#--> add your Python code here
# data_test =
# for data in data_test:
# #transform the features of the test instances to numbers following the
# same strategy done during training, and then use the decision tree to make the
# class prediction. For instance:
# #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is
# used to get an integer as the predicted class label so that you can compare it with
# the true label
# #--> add your Python code here
# #compare the prediction with the true label (located at data[3]) of the
# test instance to start calculating the model accuracy.
# #--> add your Python code here
# #find the average accuracy of this model during the 10 runs (training and
# test set)
# #--> add your Python code here
# #print the accuracy of this model during the 10 runs (training and test set).
# #your output should be something like that: final accuracy when training on
# cheat_training_1.csv: 0.2
# #--> add your Python code here