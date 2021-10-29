# Import libraries
import math
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import pickle

import warnings

# Train-Test
from sklearn.model_selection import train_test_split

# Scale
from sklearn.preprocessing import StandardScaler

# Classification Report
from sklearn.metrics import classification_report


df_19=pd.read_csv('players_19.csv')

df_19=df_19.drop(['sofifa_id','player_tags','player_traits','team_jersey_number','nation_jersey_number','loaned_from','joined',
 'contract_valid_until','player_url','long_name','age','dob','value_eur','wage_eur','height_cm','weight_kg','nationality','release_clause_eur','ls',
 'st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb',
 'cb','rcb','rb','work_rate','preferred_foot','real_face','body_type','nation_position','team_position'],axis=1)

missing_cols=['pace','shooting','passing','dribbling','defending','physic','gk_diving','gk_handling','gk_kicking','gk_reflexes','gk_speed','gk_positioning']

for i in missing_cols:
    df_19[i].fillna((0), inplace=True)

forward = ["ST", "LW", "RW", "LF", "RF", "RS","LS", "CF"]
midfielder = ["CM","RCM","LCM", "CDM","RDM","LDM", "CAM", "LAM", "RAM", "RM", "LM"]
defender = ["CB", "RCB", "LCB", "LWB", "RWB", "LB", "RB"]


for i in range(len(df_19['player_positions'])):
    df_19['player_positions'][i]=df_19['player_positions'][i].replace(' ','').split(',')[0]


df_19.loc[df_19["player_positions"] == "GK", "Position"] = 0
df_19.loc[df_19["player_positions"].isin(defender), "Position"] = 1
df_19.loc[df_19["player_positions"].isin(midfielder), "Position"] = 2
df_19.loc[df_19["player_positions"].isin(forward), "Position"] = 3

X = df_19.drop(['player_positions','Position','short_name','club'],axis=1)
y= df_19['Position']

def clean_X_values(x_value):
    cols = x_value.columns
    for col in cols:
        if(type(x_value[col][0]) == str):
            x_value[col] = [int(eval(i)) for i in x_value[col].values]
        else:
            x_value[col].fillna(np.mean(x_value[col]), inplace=True)

clean_X_values(X)


x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

lr = LogisticRegression(C=0.01,max_iter=16000)

lr.fit(x_train,y_train)

lr_y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test,lr_y_pred)


data = {"model": lr}
with open('saved_steps_2.pkl', 'wb') as file:
    pickle.dump(data, file)

with open('saved_steps_2.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded_1 = data["model"]



