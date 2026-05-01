import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
data_df = pd.read_csv("heart_failure_clinical_records_dataset.csv")


ax = sns.countplot(x=data_df["DEATH_EVENT"], palette="Set1")
ax.bar_label(ax.containers[0])

plt.show()