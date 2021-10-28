# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:01:41 2021

@author: Chetan Jagadeesh
"""

import streamlit as st
import pickle
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def load_model():
    with open('saved_steps_1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]

def clean_df(df):
    drop_cols = ['sofifa_id', 'nationality', 'team_position', 'weak_foot', \
                 'player_positions', 'preferred_foot','skill_moves', 'player_url', 'long_name', 'dob', \
                 'work_rate', 'joined', 'body_type', 'real_face', 'release_clause_eur', 'player_tags', \
                 'team_jersey_number', 'loaned_from', 'contract_valid_until', 'nation_position', \
                 'nation_jersey_number', 'player_traits', 'ls', 'st', 'rs','lw', 'lf', 'cf', 'rf', \
                 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm','rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', \
                 'rwb', 'lb', 'lcb', 'cb','rcb', 'rb']
    
    df_cleaned = df.drop(drop_cols, axis=1)
    
    X = df_cleaned.drop(['overall','short_name','club'], axis=1)
    y = df_cleaned[['overall','club','short_name']]

    for col in X.columns:
        if(type(X[col][0]) == str):
            X[col] = [int(eval(i)) for i in X[col].values]
        else:
            X[col].fillna(0, inplace=True)
    
    return X, y

def show_predict_page():
    st.title("FIFA 20 Player Overall Rating Prediction")
    from PIL import Image 
    img = Image.open("fifa 20 cover.jpg")
    st.image(img,width=400,caption='FIFA 20')


    df_20 = pd.read_csv('players_20.csv')

  


    df_X,df_y = clean_df(df_20)

    df_X[['short_name','club']]=df_20[['short_name','club']]



    teams = ['FC Barcelona','FC Bayern München','Real Madrid','Paris Saint-Germain','Juventus','Manchester City','Liverpool FC']


    team_selected=st.selectbox("Select the team down to get list of players in that club",teams)
    st.write(f"Selected club is :  {team_selected}")

    names=df_X[df_X['club']==team_selected]
    name_of_selected=names['short_name'].to_list()

    result = st.selectbox('Select the Player name',options=name_of_selected)
    st.write(f'Selected Player is:  {result}')




    club_selected = pd.DataFrame(df_X[df_X['club']==team_selected])
    player_selected=pd.DataFrame(club_selected[club_selected['short_name']==result])
    overall_of_club_selected = pd.DataFrame(df_y[df_y['club']==team_selected])
    overall_of_player_selected=overall_of_club_selected[overall_of_club_selected['short_name']==result]

    st.subheader("All the attributes of the selected player")
    #st.dataframe(player_selected)

    X=player_selected.drop(['short_name','club'],axis=1)


    y=overall_of_player_selected.drop(['short_name','club'],axis=1)


    st.dataframe(X)
    st.text("")
    st.subheader(f'The Actual Overall Rating of the {result} is   **{y.values}**')

    x_poly = PolynomialFeatures(degree=2).fit_transform(X)

    y_pred=regressor.predict(x_poly)
    st.text("")
    st.text("")
    st.write('Click on the predict button after choosing the Club and the Player.')
    if st.button('Predict the Overall Rating'):
        y_pred=regressor.predict(x_poly)
        st.text("")
        st.subheader(f'The Predicted Overall rating of {result} is **{int(np.abs(y_pred))}**')