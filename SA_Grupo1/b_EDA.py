################################################################
#               SEGUNDO TRABAJO PARA LA MATERIA:               #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE MARKETING                         #
#                              POR:                            #
#                                                              #
#                    AURA LUZ MORENO - MOOREA                  #
#                                                              #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################

#Se hace llamado  a las librerias que necesitamos:

import pandas as pd # Librería para manipulación y análisis de datos
import numpy as np # Librería para operaciones numéricas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split



# Se hace llamado al archivo de funciones
import a_funciones as fn # Archivo de funciones definidas por mi

# Queremos ver como están los datos
data = pd.read_csv("data/master.csv")
data.head()

data.shape
data.columns

#Se organzan algunos nombres que tienen caracteres especiales para su mejor manipulación
data.columns = ['country', 'year', 'gender', 'age_group', 'suicide_count', 'population', 'suicide_rate', 'country-year', 'HDI for year','gdp_for_year', 'gdp_per_capita', 'generation']
data.columns

# Que información tenemos sobre todo el dataset?
data.info()

#Podemos observar que hay una clasificación en los grupos de edades:
data.age_group.value_counts()

# age_group
# 15-24 years    4642
# 35-54 years    4642
# 75+ years      4642
# 25-34 years    4642
# 55-74 years    4642
# 5-14 years     4610

# Y también una clasificación especial acerca de esas edades
data.generation.value_counts()
# generation
# Generation X       6408
# Silent             6364
# Millenials         5844
# Boomers            4990
# G.I. Generation    2744
# Generation Z       1470


# La columna HDI for year no es clara acerca de lo que es. Y al 
# revisarlo en excel, no tiene datos. Por ende es candidato a eliminar
# Edad y generación y género son candidatos para usar un encoder
# País sería una variable para considerar si se debe usar un encoder o no. 
# Lo miraremos más adelante

country = data.country.unique()
print("Paises en la BD:", len(country))

#Lista de los 101 países en la BD
country


#Revisemos su distribución
data.hist(bins = 50,figsize = (15,11))

numeric_data = data.select_dtypes(include=[np.number])


#Revisemos el mapa de calor:
plt.figure(figsize=(7,5))
sns.heatmap(numeric_data.corr(), annot=True)
plt.show()

#Aunque ya sabemos la respuesta por la investigación previa, 
# Comprobaremos qué género presenta una mayor tasa de suicidio:

plt.figure(figsize=(10, 3))
sns.barplot(x='suicide_count', y='gender', data=data)
plt.title('Género - Suicidio')
plt.show()


#A qué edades (grupo de edad) se suicidan más los hombres?
# Aunque sabemos la respuesta, lo comprobaremos:

plt.figure(figsize=(10,3))
sns.barplot(x = "age_group", y = "suicide_count", hue = "gender", data = data)
plt.title("Rango de edad - Género")
plt.show()

#Qué generación es la que más se suicida?

plt.figure(figsize=(9,5))
sns.barplot(x = "generation", y = "suicide_count", hue = "gender", data = data)
plt.title('Generación - Género')
plt.show()

# Con esto nos queda claro que los hombres de 35-54 años, llamados "boomers" son los que más se suicidan


# Sin embargo, quiero dar una mirada más general. Sin importar el género, qué grupo de edad es el que más se suicida?

plt.figure(figsize=(9,5))
sns.barplot(x=data['age_group'], y=data['suicide_count'])
plt.xlabel('Rango de edad')
plt.ylabel('Conteo de suicidios')
plt.title('Rango de edad - Suicidios')
plt.show()


#Cuál es la generación que más incidencias de suicidio tiene?

plt.figure(figsize=(9,5))
sns.barplot(x=data['generation'], y=data['suicide_count'])
plt.xlabel('Generación')
plt.ylabel('Conteo de suicidios')
plt.title('Generación - Conteo de suicidios')
plt.show()


# Aqui un panorama más general
plt.figure(figsize=(7,7))
sns.barplot(y="gender", x="suicide_count", hue="age_group", data=data)
plt.title('Género y conteo de suicidios por rango de edad')
plt.show()

# Y por genreación
plt.figure(figsize=(7,7))
sns.barplot(y="gender", x="suicide_count", hue="generation", data=data)
plt.title('Género y Conteo de suicidios por generación')
plt.show()


# Ahora bien, como está por pais? El rango de "felicidad" sería un indicador para el más bajo?
plt.figure(figsize=(15,25))
sns.barplot(x = "suicide_rate", y = "country", data = data)
plt.title('País - Conteo de suicidios')
plt.show()

#Paises como Lituania, Hungría y Srilanka tienen una altisima tasa de suicidios.

# Este ha sido el comportamiento a través del tiempo:

data[['year','suicide_rate']].groupby(['year']).sum().plot()
#Esto podría interpretarse como que las condiciones de vida han ido mejorando, ha medida que pasan los años el conteo de suicidios es menor
#Debemos tener en cuenta que estos datos solo están hasta 2016. No encontré datos más actuales

# Pero que tantos outliers (atípicos) tenemos en la BD?
plt.figure(figsize=(20,10))
attributes = ['suicide_count', 'population', 'suicide_rate','HDI for year', 
              'gdp_for_year','gdp_per_capita']
scatter_matrix(data[attributes], figsize=(20,10))
plt.show()
#Se observan varias variables atipicas que deberán analizarse con precisión


data.describe()

#Qué datos nulos hay?  HDI for year es candidato a eliminarse.
# Se observa que tiene 19.456 datos nulos. La base de datos son 27.820. Con tanta cantidad, es mejor quitar esa columna
data.isnull().sum()

data = data.drop(['HDI for year'], axis = 1)
data.columns


data.head(10)
# La columna country-year es una combianción de dos columnas que no sonutiles. también procedemos a eliminarla

data = data.drop(['country-year'], axis = 1)
data.columns

#Quitamos los nulos
data = data.dropna()

# Y ahora haremos el encoding de todas las variables que analizamos arriba que eran susceptibles
categorical = ['country', 'year','age_group', 'gender', 'generation']
le = sklearn.preprocessing.LabelEncoder()

for column in categorical:
    data[column] = le.fit_transform(data[column])
    

#Segun lo visto en clase, haremos una copia del dataset para no dañar los datos originales
stat_data = data.copy()
stat_data

# Analicemos los datos nuevamente:

data.dtypes

# Tenemos diferencias MUY significativas entre países. Esto puede hacer que existan datos atipicos.
# Lo que queremos es estandarizarlos. Por ejemplo, la población de China con la de El Vaticano no son comparables
# Por eso es necesario escalarlas

data['gdp_for_year'] = data['gdp_for_year'].str.replace(',','').astype(float)

numerical = ['suicide_count', 'population', 'suicide_rate', 
              'gdp_for_year','gdp_per_capita']

rc = RobustScaler()
data[numerical] = rc.fit_transform(data[numerical])

data


y = data['suicide_rate']
X = data.drop('suicide_rate',axis=1)
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape