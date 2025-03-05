# -------------------------------------------------------------------------
# AUTHOR: Vi Nguyen 
# FILENAME: pca.py
# SPECIFICATION: apply PCA on given data files in this case heart_disease_dataset.csv to see the result in variation of PC1 when we 
# remove each time a distinct feature 
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 120 minutes 
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
file_path = "heart_disease_dataset.csv"
df = pd.read_csv(file_path)

#Create a training matrix without the target variable (Heart Diseas)
#--> add your Python code here
df_features = df.iloc[:, :-1]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features
#--> add your Python code here
num_features = df_features.shape[1]

pc1_var = {} 
# Run PCA for 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    reduced_data = np.delete(scaled_data, i, axis=1)

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pc1_var[df_features.columns[i]] = pca.explained_variance_ratio_[0]


# Find the maximum PC1 variance
# --> add your Python code here
max_feature = max(pc1_var, key=pc1_var.get)
max_variance = pc1_var[max_feature]
#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance: {max_variance} when remove max feature {max_feature}.")
pc1_variances_df = pd.DataFrame(list(pc1_var.items()), columns=["Feature Removed", "PC1 Var"])
print("\nPC1 Variances After Removing Each Feature:")
print(pc1_variances_df)




