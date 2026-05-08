import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None) #retira limitação de largura da tabela

#processamento de dados

data_df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
data_df.head()
data_df.info()
data_df.describe().T
data_df.isnull().sum()

X = data_df.drop(["DEATH_EVENT"], axis=1)  
y = data_df["DEATH_EVENT"]                 

s_scaler = preprocessing.StandardScaler() 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=25  
)

X_train = s_scaler.fit_transform(X_train)  
X_test = s_scaler.transform(X_test) 

#model building  
#knn
print("Evaluation for K-Nearest Neighbors".center(75, "_"))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

#knn com peso