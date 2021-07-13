# Madrid House Price Prediction Using Machine Learning
***
![Fachada Berlin](/images/fachada_verde.jpg)

## Table of contents

1. Motivation
2. Data Research
3. API request Idealista
4. wikipedia
5. munimadrid datos - criminality
6. munimadrid datos - Real State information
7. Data Preprocessing 'limpieza'
8. Modelos Pruebas
9. Final Model
10. Streamlit


Motivation
-------
__My name is Esther and I love to move from one house to another if I have the chance. During the last years I have learnt a lot aout the Real State Market in Madrid, in fact my friends and family count with my if the are looking for a new home or they are trying to sell one.__ 

__Thinking about this and how the different valiables could afect to the price (medical centers, transport, property meters,...) I found interesting to create a tool to estimate the price of a house and check how good could be.__

__Let's stat.__


Data research
-------
* The first Real State website in Spain is *Idealista*. I contacted them to get autoritation to get information from **[API](https://developers.idealista.com/access-request)** (*API_KEY,SECRET_KEY*). Access is free to a maximum of 100 req/month and it’s limited by 1 req/sec.
* [munimadrid](https://www.madrid.es/portal/site/munimadrid) website contains many resources to implement the future **ML MODEL**:
  * Something important could be is there is school centers in the area.
  * Criminality ratio.
* From [wikipedia](https://es.wikipedia.org/wiki/Wikipedia:Portada) we'll get population by district in order to get the different ratios/population. 

API request Idealista
----
Inside a .py doc calles apis.py we create a function called `def api_idealista(api_key,secret_key,file_path):` and make the request with the api_key and secret key saved in a Jupyter file called `idealista_api_call`. We will save the results. 
Idealista sent helps documentation aswell, saved on **idealista_api_doc** folder.
After that we will save the dataframe in an csv file **/Project/input/idealista.csv**

wikipedia 
-----
We use the *pandas method* `` table_MN = pd.read_html(url)``, to get the information from [wikipedia](https://es.wikipedia.org/wiki/Demograf%C3%ADa_de_Madrid) about the population in Madrid by districts. 

munimadrid datos - criminality
----
We'll get criminality information from the POLICIA MUNICIPAL in [munimadrid.es](https://datos.madrid.es/egob/catalogo/212616-89-policia-estadisticas.xlsx)
Final results will be saved in an excel file for later merge ```../Project/output/criminalidad_distrito.xls```

munimadrid datos - Real State information
----
We import a saved function ``src.apis import import_excel``` for getting excel data information about the Real State market in Madrid. This information we'll help us to check if the values we are mananing have sense.

Data Preprocessing - limpieza
----

In this Jupyter we'll load all the dataframes and make the data preprocessing, Exploratory Data Analyis and merge of the differentent dataframes. 
Finally we saved this data in an excel file for later processing of differen MODELS.
```../Project/output/project_dataset.xls```

Modelos Pruebas
------
In this Jupyter file we will make following process:
* Feature Observation.
* Feature Selecction.
* Model Building.
* Model Performances.
* Train-Test.
* Prediction.
* Final Score.
* Choose best model.

![image](https://user-images.githubusercontent.com/73304043/125522632-3f25fed4-c301-468e-8c83-89b17e587ae7.png)


Final Model
----
![image](https://user-images.githubusercontent.com/73304043/125522306-35cfa7cb-31ed-4ff6-acae-4bdfdb2b993e.png)

Output > We'll save this final model in order to import it later.

Streamlit
-----
We'll use this tool to create an app to predict the prices with the feature selection.


