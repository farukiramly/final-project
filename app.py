#import package
import streamlit as st

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


#---import data
data = pd.read_csv("Data_Clean.csv")
image = Image.open("house.jpg")
data_link= "https://www.kaggle.com/shree1992/housedata"
st.title("Welcome to House Prediction app")
st.image(image, use_column_width=True)




#checking the data
st.write("This is an application for predict how much the prices based on criteria that you choose.I take the data from [kaggle](%s)" % data_link)
check_data = st.checkbox("See the simple data")
if check_data:
    st.write(data.head())


#input the numbers sidebar
st.sidebar.title("🎊 lets choose the criteria you want🎊")
year = st.sidebar.slider("What Year Build you want?",int(data.yr_built.min()),int(data.yr_built.max()),int(data.yr_built.mean()) )
sqft_liv = st.sidebar.slider("What is your square feet of living room?",int(data.sqft_living.min()),int(data.sqft_living.max()),int(data.sqft_living.mean()) )
bath     = st.sidebar.slider("How many bathrooms?",int(data.bathrooms.min()),int(data.bathrooms.max()),int(data.bathrooms.mean()) )
bed      = st.sidebar.slider("How many bedrooms?",int(data.bedrooms.min()),int(data.bedrooms.max()),int(data.bedrooms.mean()) )
floor    = st.sidebar.slider("How many floor do you want?",int(data.floors.min()),int(data.floors.max()),int(data.floors.mean()) )



#splitting your data
X = data.drop('price', axis = 1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=45)

#modelling step
#import your model
model=LinearRegression()
#fitting and predict your model
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test))) 
predictions = model.predict([[sqft_liv,bath,bed,floor,year]])[0]

#checking prediction house price
if st.sidebar.button("See the Price! 😍"):
    st.header("Your house prices prediction is USD {}".format(int(predictions)))
    st.subheader("Your house range is USD {} - USD {}".format(int(predictions-errors),int(predictions+errors) ))

st.sidebar.subheader ('Made with 💖 by')
st.sidebar.subheader('[*Faruki Ramly*](https://www.linkedin.com/in/farukiramly/)')
#---Footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: visible;}
            footer:after{
                content:'Made with 💖 by Faruki Ramly;
                display:block;
                position:relative;
                color:tomato;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)