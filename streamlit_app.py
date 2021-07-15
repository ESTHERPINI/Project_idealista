from numpy.core.fromnumeric import size
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
import time

#progress bar
#my_bar = st.progress(0)
#for percent_complete in range(100):
    #time.sleep(0.1)
    #my_bar.progress(percent_complete + 1)

#title image
title_image = Image.open("./Project_idealista/images/fachada.jpg")
st.image(title_image)

#load dataframe
data = pd.read_excel("Project_idealista/output/project_dataset.xls")
#st.dataframe(data.head())


left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Trying to know your property price?')
if pressed:
    right_column.write("Fill in this form!")

# Headers
st.title("Madrid Real State")
st.markdown("HOW MUCH DOES YOUR HOME COST?")
st.markdown("What I love most about my home is who I share it with...")

#load final model
pickle_in = open('Project/output/KN.pkl', 'rb')
model = pickle.load(pickle_in)

## User inputs on the control panel
st.sidebar.title("PROPERTY INPUTS FOR PRICE PREDICTION")
numPhotos = st.sidebar.number_input(
    "ENTER NUMBER OF PHOTOS",
    min_value=0,
    max_value=None,
    value=1,
    step=1)

floor=st.sidebar.number_input('ENTER FLOOR NUMBER', min_value=0,
    max_value=None,
    value=1,
    step=1)

size_v = st.sidebar.number_input(
    "Sq. METERS OF THE PROPERTY",
    min_value=15,
    max_value=None,
    value=100,
    step=1,
    help="The higher this number, the expensive is the property")

rooms=st.sidebar.number_input("NUMBER OF ROOMS, LIVING ROOM NOT INCLUDED", 
    min_value=0,
    max_value=None,
    value=1,
    step=1)

bathrooms=st.sidebar.number_input("NUMBER OF BATHROOMS",
    min_value=0,
    max_value=None,
    value=1,
    step=1) 

hasLift_values=['Yes','No']
hasLift=st.sidebar.radio("HAS LIFT", hasLift_values)
dict_lift={'Yes':0,'No':1}
submit = st.button('Predict price')

train = pd.DataFrame({"numPhotos":numPhotos,'floor':floor, 'size':size_v, 'rooms':rooms,
    'bathrooms': bathrooms,'hasLift': hasLift},index=[0])        

train['hasLift']=train['hasLift'].map(dict_lift)       

if submit:
    ## Feature selection
    columnas_train = [a for a in train.columns]
    X = train[columnas_train] 
        
    prediction = model.predict(X)
    prediction=np.round(prediction,2).astype(float)
      
    st.write(f' Estimated price of the property is {prediction} € ')

#predictions_p=[]
#predictions_p.append(prediction)

numPhotos_p=[]
numPhotos_p.append(numPhotos)

floor_p=[]
floor_p.append(floor)

size_p=[]
size_p.append(size_v)

rooms_p=[]
rooms_p.append(rooms)

bathrooms_p=[]
bathrooms_p.append(bathrooms)

hasLift_p=[]
hasLift_p.append(hasLift)


df = pd.DataFrame(
        {
            #"prediction": predictions_p,
            "numPhotos":numPhotos_p,
            'floor':floor_p,
            'size':size_p,
            'rooms':rooms_p,
            'bathrooms': bathrooms_p,
            'hasLift': hasLift_p},index=[0])

#st.dataframe(df.head())


#Sidebar
st.sidebar.title("Madrid market detail")
option = st.sidebar.selectbox("which Dashboard?", ('properties by district',
'type of property on sale', 'property is exterior', 'property has parking space',"population"))

# st.header(option)

if option == 'properties by district':
    fig1 = px.histogram(data, x='district')
    fig1.update_layout(showlegend=True,
                    title="properties on sale by district",
                    title_x=0.5,
                    xaxis_title='district',
                    yaxis_title='number of properties')
    st.plotly_chart(fig1)

if option == 'type of property on sale':
    fig2 = px.histogram(data, x='propertyType')
    fig2.update_layout(showlegend=True,
                    title="type of properties on sale",
                    title_x=0.5,
                    xaxis_title='district',
                    yaxis_title='number of properties')
    st.plotly_chart(fig2)

if option == 'property has parking space':
    fig3 = px.histogram(data, x='hasParkingSpace')
    fig3.update_layout(showlegend=True,
                    title="property has parking space",
                    title_x=0.5,
                    xaxis_title='district',
                    yaxis_title='number of properties')
    st.plotly_chart(fig3)

if option == 'property is exterior':
    fig4 = px.histogram(data, x='exterior')
    fig4.update_layout(showlegend=True,
                    title="property is exterior or not",
                    title_x=0.5,
                    xaxis_title='exterior',
                    yaxis_title='number of properties')
    st.plotly_chart(fig4)

if option == 'population':
    #load dataframe
    df_population = pd.read_excel("Project_idealista/output/poblacion_distrito.xls")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_population.district,y=df_population.population,
                                 mode = 'markers',
                                 name = 'Formerly_Smoked'))
    st.plotly_chart(fig, use_container_width=True)
    



