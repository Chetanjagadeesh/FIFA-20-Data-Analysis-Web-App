# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:44:11 2021

@author: Chetan Jagadeesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#% matplotlib inline

def clean_df(df):
    drop_cols = ['sofifa_id', 'short_name', 'nationality', 'club', 'team_position', 'weak_foot', \
                 'player_positions', 'preferred_foot','skill_moves', 'player_url', 'long_name', 'dob', \
                 'work_rate', 'joined', 'body_type', 'real_face', 'release_clause_eur', 'player_tags', \
                 'team_jersey_number', 'loaned_from', 'contract_valid_until', 'nation_position', \
                 'nation_jersey_number', 'player_traits', 'ls', 'st', 'rs','lw', 'lf', 'cf', 'rf', \
                 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm','rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', \
                 'rwb', 'lb', 'lcb', 'cb','rcb', 'rb']
    
    df_cleaned = df.drop(drop_cols, axis=1)
    
    X = df_cleaned.drop('overall', axis=1)
    y = df_cleaned['overall']

    for col in X.columns:
        if(type(X[col][0]) == str):
            X[col] = [int(eval(i)) for i in X[col].values]
        else:
            X[col].fillna(0, inplace=True)
    
    return X, y


def eval_model(x, y, model):
    d = 2
    x_poly = PolynomialFeatures(degree=d).fit_transform(x)
    y_pred = model.predict(x_poly)

    mse = mean_squared_error(y, y_pred)

    score = r2_score(y, y_pred)

    return mse, score


df_19 = pd.read_csv('players_19.csv')
x,y = clean_df(df_19)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
d = 2

X_poly_train = PolynomialFeatures(degree=d).fit_transform(x_train)
X_poly_test = PolynomialFeatures(degree=d).fit_transform(x_test)



elastic_model = ElasticNet(alpha=0.01, max_iter=36000)
elastic_model.fit(X_poly_train, y_train)
y_pred_elastic = elastic_model.predict(X_poly_test)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
score_elastic = r2_score(y_test, y_pred_elastic)
print(f'Elastic Validation mse: {mse_elastic}, r2 score: {score_elastic}')


data = {"model": elastic_model}
with open('saved_steps_1.pkl', 'wb') as file:
    pickle.dump(data, file)

with open('saved_steps_1.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
#player_name = data["player_name"]



