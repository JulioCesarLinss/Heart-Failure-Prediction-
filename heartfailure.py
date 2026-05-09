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


#knn padrão

print("Evaluation for K-Nearest Neighbors".center(75, "_"))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

#knn com peso

pesos = [
    2.0, #idade
    0.5, #anemia
    0.5, #creatina fosfoquinase
    0.5, #diabetes (peso ajustado para ficar mais condizente com a realidade)
    3.0, #fração de ejeção
    1.0, #pressão alta(peso ajustado para ficar mais condizente com a realidade)
    0.5, #plaquetas
    3.0, #creatinina
    1.0, #sodio
    0.0, #sexo
    1.0, #fumante
    0.0 #tempo de acompanhamento em meses
]

#variaveis com peso

X_trainPeso = X_train * pesos 
X_testPeso = X_test * pesos


#correlacoes = data_df.corr()['DEATH_EVENT'].sort_values(ascending=False) #tabela mostrando a maior correlação entre as variáveis e o resultado final, de DEATH. 
#print(correlacoes) # É EXPLICITADO QUE A VARIÁVEL 'TIME' EXERCE GRANDE INFLUENCIA. ENTRETANTO, ELA É DESCARTÁVEL, POIS ELA É UMA TARGET LEAKAGE!!!!
#print(list(X.columns))

knn_model_pesado = KNeighborsClassifier(n_neighbors=5) #aparentemente o ideal é o 5
knn_model_pesado.fit(X_trainPeso, y_train) # alteração aqui

knn_pred_pesado = knn_model_pesado.predict(X_testPeso) # alteração aqui

knn_accuracy_pesado = accuracy_score(y_test, knn_pred_pesado)
knn_precision_pesado = precision_score(y_test, knn_pred_pesado)
knn_recall_pesado = recall_score(y_test, knn_pred_pesado)
knn_f1_pesado = f1_score(y_test, knn_pred_pesado)

#comparação na tela
print("\n--- COMPARAÇÃO DE ACURÁCIA ---")
print(f"KNN Padrão: {knn_accuracy:.4f}")
print(f"KNN com Pesos Customizados: {knn_accuracy_pesado:.4f}")

print("\n--- COMPARAÇÃO DE RECALL (Capacidade de detectar risco) ---")
print(f"KNN Padrão - Recall: {knn_recall:.4f} | F1: {knn_f1:.4f}")
print(f"KNN com Pesos - Recall: {knn_recall_pesado:.4f} | F1: {knn_f1_pesado:.4f}")


