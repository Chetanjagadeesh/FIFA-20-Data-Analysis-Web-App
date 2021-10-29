import streamlit as st
from app import show_predict_page
from player_position import show_position_prediction


st.sidebar.title("FIFA 20 Data Analysis")

#st.sidebar.write('Football is arguably the most popular sport in the world and FIFA is the most popular football (soccer) simulation game by Electronic Arts (EA Sports).')
#st.sidebar.markdown("""---""")
st.sidebar.write('Predicting **Player Overall Rating ** and **Player Position** on **FIFA 20** dataset using a Machine Learning Model built on **FIFA 19** dataset')
 
st.sidebar.markdown("""---""")

page = st.sidebar.radio("Click on the button below to perform the task", ("Predict Overall Rating", "Predict Player Position"))

if page == "Predict Overall Rating":
    show_predict_page()
else:
    show_position_prediction()

st.sidebar.markdown("""---""")

st.sidebar.header('AI-1 Project: Let's Play FIFA⚽')
st.sidebar.subheader('1.Chetan Jagadeesh')
st.sidebar.subheader('2.Krishna Chaitanya Velaga')
st.sidebar.subheader('3.Nivas Ramisetty')
st.sidebar.subheader('4.Sahil Joshi')
