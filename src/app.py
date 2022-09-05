import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from collections import Counter

df_raw = pd.read_csv('../data/raw/healthcare-dataset-stroke-data.csv') 
 
#Impute BMI value
labels = ['0-34.9','35 - 44.9', '45 - 54.9', '55 - 64.9', '65 - +'] 
# Define the edges between bins
bins = [0,35, 45, 55, 65, np.inf]
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
 
# Encoding the 'smoking status' column
df_raw['smoking_status'] = df_raw['smoking_status'].map({'Unknown': 0, 'never smoked' : 1, 'smokes': 2 , 'formerly smoked':3})

#drop outliers bmi
df_raw.drop(df_raw[df_raw['bmi']>80].index,inplace=True)

#Remove features
df_raw.drop(["age_bins","bmi_new","id"],axis=1,inplace=True)
df_raw.drop(["ever_married","work_type","Residence_type","heart_disease"],axis=1,inplace=True)
 
#Scaler the float data
#scaler = MinMaxScaler()
#train_scaler = scaler.fit(df_raw[['age','bmi','avg_glucose_level']])
#df_raw[['age','bmi','avg_glucose_level']] = train_scaler.transform(df_raw[['age','bmi','avg_glucose_level']])

#we define our labels and features
y = df_raw['stroke']
X = df_raw.drop('stroke', axis=1)
#we divide into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13, stratify=y)

def run_model_balanced(X_train, X_test, y_train, y_test, weight={1:18,0:1}):
    clf = LogisticRegression(C=0.1,penalty='l2',random_state=1,solver="newton-cg",class_weight=weight)
    clf.fit(X_train, y_train)
    return clf
 
model_balanced = run_model_balanced(X_train, X_test, y_train, y_test) 
pred_y = model_balanced.predict(X_test)
print(confusion_matrix(y_test, pred_y))
print(classification_report(y_test, pred_y,zero_division=False))

print(X_test.sample(1))


#Flask Dump
filename = '../models/stroke_model.pkl'
pickle.dump(model_balanced, open(filename,'wb'))

"""
def run_model(X_train, X_test, y_train, y_test):
    clf_base = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg")
    clf_base.fit(X_train, y_train)
    return clf_base

y_train_nostroke=y_train[y_train==0].sample(1600)
list(y_train_nostroke.index)
X_train_nostroke=X_train[X_train.index.isin(list(y_train_nostroke.index))]
len(X_train_nostroke)
X_train_muestra=X_train[~ X_train.index.isin(list(y_train_nostroke.index))]
y_train_muestra= y_train[~ y_train.index.isin(list(y_train_nostroke.index))]
y_train_muestra.value_counts()

from imblearn.over_sampling import RandomOverSampler

os =  RandomOverSampler()
X_train_res, y_train_res = os.fit_resample(X_train_muestra, y_train_muestra)

print ("Distribution before resampling {}".format(Counter(y_train_muestra)))
print ("Distribution labels after resampling {}".format(Counter(y_train_res)))

y_train_final=pd.concat([y_train_nostroke, y_train_res ], ignore_index=True)
X_train_final=pd.concat([X_train_nostroke, X_train_res ], ignore_index=True)

print ("Distribution labels finals {}".format(Counter(y_train_final)))

model_samp = run_model(X_train_final, X_test, y_train_final, y_test) 
pred_y = model_samp.predict(X_test)
print(confusion_matrix(y_test, pred_y))
print(classification_report(y_test, pred_y,zero_division=False))
"""