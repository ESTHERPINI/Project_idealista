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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as pylab
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix , accuracy_score, precision_score, recall_score
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
import sklearn.metrics as metrics
from sklearn.linear_model import BayesianRidge


#progress bar
#my_bar = st.progress(0)
#for percent_complete in range(100):
    #time.sleep(0.1)
    #my_bar.progress(percent_complete + 1)


# Create a page dropdown 
st.title("APP PAGES")
page = st.radio("SELECT", ["Features and models", "Final model prediction"]) 



if page == "Final model prediction":
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


    
if page == "Features and models":
    # Display details of page 2

    st.title('ML MODELS FOR HOUSE PRICE PREDICTION')
    #title image
    title_image = Image.open("./Project_idealista/images/BRAIN.jpg")
    st.image(title_image)


    #load dataframe
    df = pd.read_excel("Project_idealista/input/df_train.xls")
    df.drop(columns = ['Unnamed: 0'], inplace = True)

    data = pd.read_excel("Project_idealista/output/project_dataset.xls")
    #st.dataframe(data.head())

    #data2=df.groupby(['district'])[['price','size','bathrooms','rooms']].mean().reset_index()
    #st.write('Mean prices, size, bathrooms, rooms by district',data2)


    # Allow use to choose sidebar
    st.sidebar.title("FEATURES EXPLORE")
    option = st.sidebar.selectbox("which value?", ('size vs. price','rooms vs. bathrooms','numPhotos', 'floor', 'size', 'rooms', 'bathrooms', 'hasLift'))

    

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

    if option == 'rooms vs. bathrooms':
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(df["rooms"],df["bathrooms"])
        ax.set_xlabel("rooms of the property")
        ax.set_ylabel("bathrooms")
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

    alg = ['KneighborsRegressor','LinearRegression','Ridge', 'Bayesian','Lasso']


    predictor = st.selectbox('Which algorithm?', alg)



    #Linear Regression
    if predictor=='LinearRegression':
        LR=LinearRegression()
        LR.fit(X_train, y_train)
        y_pred=LR.predict(X_train) 

        testing_score=[]
        training_score=[]
        cv_results_rms = []
        R2=[]
        adj_R2=[]
        MAE=[]
        MSE=[]
        RMSE=[]   

        #TRAINING ACC SCORE
        LR_traing_score=LR.score(X_train, y_train)*100
        training_score.append(LR_traing_score.mean())
        st.write("Training Accuracy:", LR.score(X_train,y_train)*100)
        #TESTING ACC SCORE
        LR_testing_score=LR.score(X_test, y_test)*100
        testing_score.append(LR_testing_score.mean())
        st.write("Testing Accuracy:", LR.score(X_test, y_test)*100)



        #X_test predict
        cv_score = cross_val_score(LR, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
        cv_results_rms.append(cv_score.mean())
        st.write("cross_val ","%s: %f " % ('LR', cv_score.mean()))

        pred = LR.predict(X_test)

        #R2
        R2_metrics=metrics.r2_score(y_test, pred)
        R2.append(R2_metrics.mean())
        st.write("R2 ","%s: %f " % ('LR',  R2_metrics.mean()))

        #adj_R2
        adj_R2_metrics=(1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
        adj_R2.append(adj_R2_metrics.mean())
        st.write("adj_R2 ","%s: %f " % ('LR',  adj_R2_metrics.mean()))   

        #MAE
        MAE_metrics=metrics.mean_absolute_error(y_test, pred)
        MAE.append(MAE_metrics.mean())
        st.write("MAE", "%s: %f " % ('LR',  MAE_metrics.mean()))  

        #MSE    
        MSE_metrics=metrics.mean_squared_error(y_test, pred)
        MSE.append(MSE_metrics.mean())
        st.write("MSE ","%s: %f " % ('LR',  MSE_metrics.mean()))

        #RMSE
        RMSE_metrics=np.sqrt(metrics.mean_squared_error(y_test, pred))
        RMSE.append(RMSE_metrics.mean())
        st.write("RMSE ","%s: %f " % ('LR', RMSE_metrics.mean())) 

        df_random = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})
        df_random = df_random.sort_values('Actual')
        st.write(df_random.head())
        #Actual	vs Predicted
        
                
        st.header('Actual vs. Predicted prices- scatter & bars')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_random.Actual,y=df_random.Predicted,
                                    mode = 'markers',
                                    name = 'Formerly_Smoked'))
        st.plotly_chart(fig, use_container_width=True)


        df_random_plot=df_random.sample(n=30, random_state=1)
        st.line_chart(df_random_plot)




    #RIDGE
    if predictor=='Ridge':
        ridge=Ridge(alpha=0.5)
        ridge.fit(X_train, y_train)
        y_pred=ridge.predict(X_train) 

        testing_score=[]
        training_score=[]
        cv_results_rms = []
        R2=[]
        adj_R2=[]
        MAE=[]
        MSE=[]
        RMSE=[] 

        ridge_traing_score=ridge.score(X_train, y_train)*100
        training_score.append(ridge_traing_score.mean())
        st.write("Training Accuracy: ", ridge.score(X_train , y_train)*100)

        ridge_testing_score=ridge.score(X_test, y_test)*100
        testing_score.append(ridge_testing_score.mean())
        st.write("Testing Accuracy:" , ridge.score(X_test, y_test)*100)

        cv_score = cross_val_score(ridge, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
        cv_results_rms.append(cv_score.mean())
        st.write("cross_val ","%s: %f " % ('ridge', cv_score.mean()))

        pred = ridge.predict(X_test)

        #R2
        R2_metrics=metrics.r2_score(y_test, pred)
        R2.append(R2_metrics.mean())
        st.write("R2 ","%s: %f " % (ridge,  R2_metrics.mean()))

        #adj_R2
        adj_R2_metrics=(1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
        adj_R2.append(adj_R2_metrics.mean())
        st.write("adj_R2 ","%s: %f " % (ridge,  adj_R2_metrics.mean()))   

        #MAE
        MAE_metrics=metrics.mean_absolute_error(y_test, pred)
        MAE.append(MAE_metrics.mean())
        st.write("MAE", "%s: %f " % (ridge,  MAE_metrics.mean()))  

        #MSE    
        MSE_metrics=metrics.mean_squared_error(y_test, pred)
        MSE.append(MSE_metrics.mean())
        st.write("MSE ","%s: %f " % (ridge,  MSE_metrics.mean()))

        #RMSE
        RMSE_metrics=np.sqrt(metrics.mean_squared_error(y_test, pred))
        RMSE.append(RMSE_metrics.mean())
        st.write("RMSE ","%s: %f " % (ridge, RMSE_metrics.mean()))    

        df_random = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})
        df_random = df_random.sort_values('Actual')
        st.write(df_random.head())
        #Actual	vs Predicted
        
        st.header('Actual vs. Predicted prices- scatter & bars')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_random.Actual,y=df_random.Predicted,
                                    mode = 'markers',
                                    name = 'Formerly_Smoked'))
        st.plotly_chart(fig, use_container_width=True)


        df_random_plot=df_random.sample(n=30, random_state=1)
        st.line_chart(df_random_plot)

    if predictor=='Lasso':
        
        lasso = Lasso(alpha = 0.01)
        lasso.fit(X_train, y_train)
        y_pred=lasso.predict(X_train)

        testing_score=[]
        training_score=[]
        cv_results_rms = []
        R2=[]
        adj_R2=[]
        MAE=[]
        MSE=[]
        RMSE=[] 
        
        lasso_training_score=lasso.score(X_train, y_train)*100
        training_score.append(lasso_training_score.mean())
        st.write("Training Accuracy: ", lasso.score(X_train , y_train)*100)

        lasso_testing_score=lasso.score(X_test, y_test)*100
        testing_score.append(lasso_testing_score.mean())
        st.write("Testing Accuracy:" , lasso.score(X_test, y_test)*100)

        cv_score = cross_val_score(lasso, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
        cv_results_rms.append(cv_score.mean())
        st.write("cross_val ","%s: %f " % ('Lasso', cv_score.mean()))

        pred = lasso.predict(X_test)

        #R2
        R2_metrics=metrics.r2_score(y_test, pred)
        R2.append(R2_metrics.mean())
        st.write("R2 ","%s: %f " % (lasso,  R2_metrics.mean()))

        #adj_R2
        adj_R2_metrics=(1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
        adj_R2.append(adj_R2_metrics.mean())
        st.write("adj_R2 ","%s: %f " % (lasso,  adj_R2_metrics.mean()))   

        #MAE
        MAE_metrics=metrics.mean_absolute_error(y_test, pred)
        MAE.append(MAE_metrics.mean())
        st.write("MAE", "%s: %f " % (lasso,  MAE_metrics.mean()))  

        #MSE    
        MSE_metrics=metrics.mean_squared_error(y_test, pred)
        MSE.append(MSE_metrics.mean())
        st.write("MSE ","%s: %f " % (lasso,  MSE_metrics.mean()))

        #RMSE
        RMSE_metrics=np.sqrt(metrics.mean_squared_error(y_test, pred))
        RMSE.append(RMSE_metrics.mean())
        st.write("RMSE ","%s: %f " % (lasso, RMSE_metrics.mean()))  

        df_random = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})
        df_random = df_random.sort_values('Actual')
        st.write(df_random.head())
        #Actual	vs Predicted
        
        st.header('Actual vs. Predicted prices- scatter & bars')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_random.Actual,y=df_random.Predicted,
                                    mode = 'markers',
                                    name = 'Formerly_Smoked'))
        st.plotly_chart(fig, use_container_width=True)


        df_random_plot=df_random.sample(n=30, random_state=1)
        st.line_chart(df_random_plot)

    if predictor=='KneighborsRegressor':     
        KN = KNeighborsRegressor()
        KN.fit(X_train, y_train)
        y_pred=KN.predict(X_train)  

        testing_score=[]
        training_score=[]
        cv_results_rms = []
        R2=[]
        adj_R2=[]
        MAE=[]
        MSE=[]
        RMSE=[]
        
        KN_training_score=KN.score(X_train , y_train)*100
        training_score.append(KN_training_score.mean())
        st.write("Training Accuracy: ", KN.score(X_train , y_train)*100)

        KN_testing_score=KN.score(X_train , y_train)*100
        testing_score.append(KN_testing_score.mean())
        st.write("Testing Accuracy:" , KN.score(X_test, y_test)*100)

        cv_score = cross_val_score(KN, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
        cv_results_rms.append(cv_score.mean())
        st.write("cross_val ","%s: %f " % ('KN', cv_score.mean()))

        pred =KN.predict(X_test)

        #R2
        R2_metrics=metrics.r2_score(y_test, pred)
        R2.append(R2_metrics.mean())
        st.write("R2 ","%s: %f " % (KN,  R2_metrics.mean()))

        #adj_R2
        adj_R2_metrics=(1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
        adj_R2.append(adj_R2_metrics.mean())
        st.write("adj_R2 ","%s: %f " % (KN,  adj_R2_metrics.mean()))   

        #MAE
        MAE_metrics=metrics.mean_absolute_error(y_test, pred)
        MAE.append(MAE_metrics.mean())
        st.write("MAE", "%s: %f " % (KN,  MAE_metrics.mean()))  

        #MSE    
        MSE_metrics=metrics.mean_squared_error(y_test, pred)
        MSE.append(MSE_metrics.mean())
        st.write("MSE ","%s: %f " % (KN,  MSE_metrics.mean()))

        #RMSE
        RMSE_metrics=np.sqrt(metrics.mean_squared_error(y_test, pred))
        RMSE.append(RMSE_metrics.mean())
        st.write("RMSE ","%s: %f " % (KN, RMSE_metrics.mean()))   

        df_random = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})
        df_random = df_random.sort_values('Actual')
        st.write(df_random.head())
        #Actual	vs Predicted
        
        st.header('Actual vs. Predicted prices- scatter & bars')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_random.Actual,y=df_random.Predicted,
                                    mode = 'markers',
                                    name = 'Formerly_Smoked'))
        st.plotly_chart(fig, use_container_width=True)


        df_random_plot=df_random.sample(n=30, random_state=1)
        st.line_chart(df_random_plot) 

    if predictor=='Bayesian':     
        bayesian = BayesianRidge()
        bayesian.fit(X_train, y_train)
        y_pred=bayesian.predict(X_train)  

        testing_score=[]
        training_score=[]
        cv_results_rms = []
        R2=[]
        adj_R2=[]
        MAE=[]
        MSE=[]
        RMSE=[]
        
        bayesian_training_score=bayesian.score(X_train, y_train)*100
        training_score.append(bayesian_training_score.mean())
        st.write("Training Accuracy: ", bayesian.score(X_train , y_train)*100)

        bayesian_testing_score=bayesian.score(X_test, y_test)*100
        testing_score.append(bayesian_testing_score.mean())
        st.write("Testing Accuracy:" , bayesian.score(X_test, y_test)*100)

        cv_score = cross_val_score(bayesian, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
        cv_results_rms.append(cv_score.mean())
        st.write("cross_val ","%s: %f " % ('bayesian', cv_score.mean()))

        pred = bayesian.predict(X_test)

        #R2
        R2_metrics=metrics.r2_score(y_test, pred)
        R2.append(R2_metrics.mean())
        st.write("R2 ","%s: %f " % (bayesian,  R2_metrics.mean()))

        #adj_R2
        adj_R2_metrics=(1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
        adj_R2.append(adj_R2_metrics.mean())
        st.write("adj_R2 ","%s: %f " % (bayesian,  adj_R2_metrics.mean()))   

        #MAE
        MAE_metrics=metrics.mean_absolute_error(y_test, pred)
        MAE.append(MAE_metrics.mean())
        st.write("MAE", "%s: %f " % (bayesian,  MAE_metrics.mean()))  

        #MSE    
        MSE_metrics=metrics.mean_squared_error(y_test, pred)
        MSE.append(MSE_metrics.mean())
        st.write("MSE ","%s: %f " % (bayesian,  MSE_metrics.mean()))

        #RMSE
        RMSE_metrics=np.sqrt(metrics.mean_squared_error(y_test, pred))
        RMSE.append(RMSE_metrics.mean())
        st.write("RMSE ","%s: %f " % (bayesian, RMSE_metrics.mean()))  

        df_random = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})
        df_random = df_random.sort_values('Actual')
        st.write(df_random.head())
        #Actual	vs Predicted
        
        st.header('Actual vs. Predicted prices- scatter & bars')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_random.Actual,y=df_random.Predicted,
                                    mode = 'markers',
                                    name = 'Formerly_Smoked'))
        st.plotly_chart(fig, use_container_width=True)


        df_random_plot=df_random.sample(n=30, random_state=1)
        st.line_chart(df_random_plot)  


      
        
    #Sidebar
    st.sidebar.title("Madrid market detail")
    option = st.sidebar.selectbox("which Dashboard?", ('properties by district',
    'type of property on sale', 'property is exterior', 'property has parking space',"population"))

    st.title('MADRID MARKET')

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
        


    