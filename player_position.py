import streamlit as st
import pickle
import numpy as np
import requests
import io 
import PIL
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def load_model():
    with open('saved_steps_2.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]

def clean_X_values(x_value):
    cols = x_value.columns
    for col in cols:
        if(type(x_value[col][0]) == str):
            x_value[col] = [int(eval(i)) for i in x_value[col].values]
        else:
            x_value[col].fillna(np.mean(x_value[col]), inplace=True)

def position(y):
    if y==0:
        return 'Goalkeeper'
    elif y==1:
        return 'Defender'
    elif y==2:
        return 'Midfielder'
    else:
        return 'Forward'


def show_position_prediction():

    st.title('⚽🏆FIFA 20 Player Position Prediction')

    st.write("Using player skill attributes we are predicting whether the player plays the following positions:")
    st.write("1.**Goalkeeper**")
    st.write("2.**Defender**")
    st.write("3.**Midfielder**")
    st.write("4.**Forward**")

    from PIL import Image 
    img = Image.open("world 11 fifa cover.jpg")
    st.image(img,width=500,caption='World 11')


    df_20=pd.read_csv('players_20.csv')

    df_20=df_20.drop(['sofifa_id','player_tags','player_traits','team_jersey_number','nation_jersey_number','loaned_from','joined','contract_valid_until','player_url','long_name','age','dob','value_eur','wage_eur','height_cm','weight_kg','nationality','release_clause_eur','ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb','work_rate','preferred_foot','real_face','body_type','nation_position','team_position'],axis=1)

    missing_cols=['pace','shooting','passing','dribbling','defending','physic','gk_diving','gk_handling','gk_kicking','gk_reflexes','gk_speed','gk_positioning']

    for i in missing_cols:
        df_20[i].fillna((0), inplace=True)
    
    forward = ["ST", "LW", "RW", "LF", "RF", "RS","LS", "CF"]
    midfielder = ["CM","RCM","LCM", "CDM","RDM","LDM", "CAM", "LAM", "RAM", "RM", "LM"]
    defender = ["CB", "RCB", "LCB", "LWB", "RWB", "LB", "RB"]


    for i in range(len(df_20['player_positions'])):
        df_20['player_positions'][i]=df_20['player_positions'][i].replace(' ','').split(',')[0]


    df_20.loc[df_20["player_positions"] == "GK", "Position"] = 0
    df_20.loc[df_20["player_positions"].isin(defender), "Position"] = 1
    df_20.loc[df_20["player_positions"].isin(midfielder), "Position"] = 2
    df_20.loc[df_20["player_positions"].isin(forward), "Position"] = 3

    teams=['FC Barcelona','FC Bayern München','Real Madrid','Paris Saint-Germain','Juventus','Atlético Madrid','Manchester City','Liverpool','Manchester United']
    
    team_selected=st.selectbox("Select the team down to get list of players in that club",teams)
    st.write(f"Selected club is :  {team_selected}")

    names=df_20[df_20['club']==team_selected]
    name_of_selected=names['short_name'].to_list()

    result = st.selectbox('Select the Player name',options=name_of_selected)
    st.write(f'Selected Player is:  {result}')

    club_selected = pd.DataFrame(df_20[df_20['club']==team_selected])
    player_selected=pd.DataFrame(club_selected[club_selected['short_name']==result])

    photo_dataset=pd.read_csv('fifa20_data.csv')
    player_photo_team = photo_dataset[photo_dataset['Club']==team_selected]
    player_photo_selected=player_photo_team[player_photo_team['Short_Name'].str.lstrip()==result]
    player_photo_url=player_photo_selected['Image']
 
    response = requests.get(player_photo_url.iloc[0])
    image_bytes = io.BytesIO(response.content)

    img = PIL.Image.open(image_bytes)
    
    st.image(img,width=200,caption=result)

    st.subheader("💪All the attributes of the selected player")

    y=player_selected['Position']

    X=player_selected.drop(['short_name','club','player_positions','Position'],axis=1)

    

    st.dataframe(X)
    st.text("")


    st.subheader(f'🥇The Actual Position of {result} is   **{position(y.values)}**')

    y_pred=regressor.predict(X)
    st.text("")
    st.text("")
    st.write('Click on the predict button after choosing the Club and the Player.')
    if st.button('👉Predict the Player Position'):
        y_pred=regressor.predict(X)
        st.text("")
        st.subheader(f'The Predicted Position {result} is **{position(int(np.abs(y_pred)))}**')

    




    
