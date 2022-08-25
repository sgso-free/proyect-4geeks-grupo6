import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from imblearn.under_sampling import NearMiss
from collections import Counter


df_raw = pd.read_csv('/content/drive/MyDrive/4geeks/Proyecto/colab/healthcare-dataset-stroke-data.csv')

#Remove data about person below 35 years-old
df_filter_age35 = df_raw[df_raw['age']>35]
df_raw = df_filter_age35.copy()

#Impute BMI value
labels = ['35 - 44.9', '45 - 54.9', '55 - 64.9', '65 - +'] 
# Define the edges between bins
bins = [35, 45, 55, 65, np.inf]

# pd.cut each column, with each bin closed on left and open on right
df_raw['age_bins'] = pd.cut(df_raw['age'], bins=bins, labels=labels, right=False)
#set bmi using the mean of each range of age
df_raw['bmi_new'] = df_raw.groupby("age_bins")['bmi'].transform(lambda x: x.fillna(x.mean()))
df_raw['bmi'].fillna(df_raw['bmi_new'], inplace = True)

#set age as int
df_raw['age']=df_raw['age'].astype(int)

# Encoding the 'Sex' column
df_raw['gender'] = df_raw['gender'].apply(lambda x: 1 if x == 'Male' else 0)
df_raw['gender'].astype(int)

# Encoding the 'Residence_type' column
df_raw['Residence_type'] = df_raw['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

# Encoding the 'Residence_type' column
df_raw['ever_married'] = df_raw['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
 
# Encoding the 'work type' column
df_raw['work_type'] = df_raw['work_type'].map({'Private' : 0, 'Self-employed': 1, 'children': 2 , 'Govt_job':3, 'Never_worked':4})

# Encoding the 'smoking status' column
df_raw['smoking_status'] = df_raw['smoking_status'].map({'Unknown': 0, 'never smoked' : 1, 'smokes': 2 , 'formerly smoked':3})

#Remove features
df_raw.drop(["age_bins","bmi_new","id"],axis=1,inplace=True)

