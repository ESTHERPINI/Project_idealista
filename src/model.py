import pandas as pd  
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
#from sklearn import metric
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix , accuracy_score, precision_score, recall_score
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso

st.title('ML MODELS FOR HOUSE PRICE PREDICTION')
#title image
title_image = Image.open("./Project_idealista/images/KN.jpeg")
st.image(title_image)


#load dataframe
df = pd.read_excel("Project_idealista/input/df_train.xls")
df.drop(columns = ['Unnamed: 0'], inplace = True)

#data2=df.groupby(['district'])[['price','size','bathrooms','rooms']].mean().reset_index()
#st.write('Mean prices, size, bathrooms, rooms by district',data2)

# Allow use to choose sidebar
st.sidebar.title("FEATURES EXPLORE")
option = st.sidebar.selectbox("which value?", ('size vs. price','numPhotos', 'floor', 'size', 'rooms', 'bathrooms', 'hasLift'))

# st.header(option)

if option == 'numPhotos':
    fig1 = px.histogram(df, x='numPhotos')
    fig1.update_layout(showlegend=True,
                    title="Number of pictures on advertisement",
                    title_x=0.5,
                    xaxis_title='Pictures',
                    yaxis_title='number of properties')
    st.plotly_chart(fig1)

if option == 'floor':
    fig1 = px.histogram(df, x='floor')
    fig1.update_layout(showlegend=True,
                    title="Number of floor of the advertisement",
                    title_x=0.5,
                    xaxis_title='Floor',
                    yaxis_title='number of properties')
    st.plotly_chart(fig1)

if option == 'rooms':
    fig1 = px.histogram(df, x='rooms')
    fig1.update_layout(showlegend=True,
                    title="Rooms of the property",
                    title_x=0.5,
                    xaxis_title='rooms',
                    yaxis_title='number of properties')
    st.plotly_chart(fig1)

if option == 'size':
    fig1 = px.histogram(df, x='size')
    fig1.update_layout(showlegend=True,
                    title="Size of the property",
                    title_x=0.5,
                    xaxis_title='size',
                    yaxis_title='number of properties')
    st.plotly_chart(fig1)

if option == 'bathrooms':
    fig1 = px.histogram(df, x='bathrooms')
    fig1.update_layout(showlegend=True,
                    title="Bathrooms of the property",
                    title_x=0.5,
                    xaxis_title='bathrooms',
                    yaxis_title='bathrooms of properties')
    st.plotly_chart(fig1)

if option == 'hasLift':
    fig1 = px.histogram(df, x='hasLift')
    fig1.update_layout(showlegend=True,
                    title="Property has Lift",
                    title_x=0.5,
                    xaxis_title='Lift or not',
                    yaxis_title='number of properties')
    st.plotly_chart(fig1)

if option == 'size vs. price':
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(df["size"],df["price"])
    ax.set_xlabel("properties size")
    ax.set_ylabel("prices")
    st.write(fig)



# Model
st.header('Feature Selection')

columnas_train = [a for a in df.columns if a not in ["price"]]
X = df[columnas_train] #independent columns
y = np.round(df['price']) # target column, to be sure that the variable has int values
st.write(X)
st.write(y)


st.header('Model Fitting')

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=7)

alg = ['KneighborsRegressor','LinearRegression','Ridge']
predictor = st.selectbox('Which algorithm?', alg)


#Linear Regression
if predictor=='LinearRegression':
    LR=LinearRegression()
    LR.fit(X_train, y_train)
    y_pred=LR.predict(X_train)    
    cm=confusion_matrix(y_test,y_pred)
    st.write('Confusion matrix: ', cm)

testing_score=[]
training_score=[]
cv_results_rms = []
R2=[]
adj_R2=[]
MAE=[]
MSE=[]
RMSE=[]

LR_training_score=LR.score(X_train , y_train)*100
training_score.append(LR_training_score.mean())
st.write("Training Accuracy: ", LR.score(X_train , y_train)*100)

LR_testing_score=LR.score(X_train , y_train)*100
testing_score.append(LR_testing_score.mean())
st.write("Testing Accuracy:" , LR.score(X_test, y_test)*100)

cv_score = cross_val_score(LR, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
cv_results_rms.append(cv_score.mean())
st.write("cross_val ","%s: %f " % (LR, cv_score.mean()))

pred =LR.predict(X_test)

#R2
R2_metrics=r2_score(y_test, pred)
R2.append(R2_metrics.mean())
st.write("R2 ","%s: %f " % (LR,  R2_metrics.mean()))

#MAE
#MAE_metrics=mean_absolute_error(y_test, pred)
#MAE.append(MAE_metrics.mean())
#st.write("MAE", "%s: %f " % (KN,  MAE_metrics.mean()))  

#MSE    
MSE_metrics=mean_squared_error(y_test, pred)
MSE.append(MSE_metrics.mean())
st.write("MSE ","%s: %f " % (LR,  MSE_metrics.mean()))

#RMSE
#If True returns MSE value, if False returns RMSE value.
#RMSE_metrics=st.write(mean_squared_error(y_test, pred, squared=False))
#RMSE.append(RMSE_metrics.mean())
#st.write("RMSE ","%s: %f " % (KN, RMSE_metrics.mean()))  



