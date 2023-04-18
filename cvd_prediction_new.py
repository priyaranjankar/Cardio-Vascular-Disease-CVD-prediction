# -*- coding: utf-8 -*-
"""cvd_prediction_new.ipynb
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

"""#  Importing Data"""

raw_data = pd.read_csv("https://raw.githubusercontent.com/priyaranjankar/Cardio-Vascular-Disease-CVD-prediction/main/cardio_train.csv",
                      delimiter=';')

raw_data.head()

"""#  Data Preprocessing

Checking null values.
"""

raw_data.info

raw_data.isna().sum()

"""As no null values found, proceeding ahead with pre-processing steps.
1. Here ID column is not required in our operations. Hence removing it.
2. We also need to check the datatypes of the features.
3. One hot encode the categorical features
4. Separate the target variable from the training features.
5. Normalizing the predictor variables

1. Dropping Id column
"""

raw_data.drop(['id'], axis=1, inplace=True)
raw_data.columns

"""2. Treating datatypes of all attributes"""

print(raw_data.dtypes)

# Converting age from days to years
raw_data['age'] = (raw_data['age'] / 365).round().astype(int)
raw_data.head()

"""3. One hot Encoding categorical variables """

df = pd.get_dummies(raw_data, columns=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
df.head()

"""4. separating target variable from training features - 'cardio'"""

X = df.drop(['cardio'], axis = 1)
y = df['cardio']
print(X.shape)
print(y.shape)

"""5. Normalizing predictor variables"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

"""# Train-Test Split"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 81)

X_train.shape

X_test.shape

"""# **MODELLING & EVALUATION**"""

!pip install xgboost

# Import necessary libraries
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Define a list of models to try
models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier(),
    MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
]

# Define a dictionary to store the cross-validation results for each model
results = {}

# Perform 5-fold cross-validation for each model and store the results in the dictionary
for model in models:
    scores = cross_validate(model, X_train, y_train, cv=5, scoring='accuracy')
    results[str(model)] = np.mean(scores['test_score'])

# Select the best model based on the cross-validation results
best_model = max(results, key=results.get)
best_model = eval(best_model)

# Train the best model on the full training set
best_model.fit(X_train, y_train)

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the performance of the best model using metrics such as accuracy, precision, recall, and AUC-ROC
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)

print("Best model: ", best_model)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("AUC-ROC: ", auc_roc)

"""Best model: GradientBoostingClassifier() - This indicates the type of model that was found to have the best performance according to the chosen evaluation metric, in this case, cross-validation accuracy.

Accuracy: 0.7323571428571428 - This represents the overall accuracy of the model in predicting the correct class label for each observation in the dataset.

Precision: 0.7496516488620529 - Precision measures how many of the predicted positive instances are actually positive. It is the ratio of true positives to the sum of true positives and false positives.

Recall: 0.6944922547332186 - Recall, also known as sensitivity or true positive rate, measures how many of the actual positive instances were correctly predicted by the model. It is the ratio of true positives to the sum of true positives and false negatives.

AUC-ROC: 0.7322062867291591 - The area under the receiver operating characteristic (ROC) curve is a measure of how well the model can distinguish between the positive and negative classes. It summarizes the trade-off between sensitivity and specificity. An AUC-ROC score of 1 indicates perfect performance, while a score of 0.5 indicates random guessing.

# **MODEL DEPLOYMENT**
"""

import pickle

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
