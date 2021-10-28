import streamlit as st
from app import show_predict_page
from player_position import show_position_prediction

st.sidebar.title("FIFA 20 Data Analysis")
page = st.sidebar.radio("Click on the button below to perform the task", ("Predict Overall Rating", "Predict Player Position"))

if page == "Predict Overall Rating":
    show_predict_page()
else:
    show_position_prediction()