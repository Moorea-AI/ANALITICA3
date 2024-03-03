################################################################
#               PRIMER TRABAJO PARA LA MATERIA:                #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE RECURSOS HUMANOS                  #
#                              POR:                            #
#                                                              #
#                    AURA LUZ MORENO - MOOREA                  #
#                       ALEJANDRA AGUIRRE                      #
#                                                              #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################

#### Cargar paquetes siempre al inicio
import sqlite3 as sql #### para bases de datos sql
import a_funciones as a_funciones ### archivo de funciones propias
import seaborn as sns
import numpy as np 
import graphviz

import pandas as pd ### para manejo de datos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones

import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos

from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import export_text ## para exportar reglas del árbol
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


conn = sql.connect("db_empleados") 
df = pd.read_sql("SELECT * FROM all_employees", conn)
df.drop('index', axis = 1, inplace = True)  
#Borramos la columna Index ya que no aporta nada relevante

# Define transformaciones para variables numéricas y categóricas
numeric_features = ['Age', 'JobSatisfaction', ...]  # Variables numéricas
categorical_features = ['Education', 'Department', ...]  # Variables categóricas

numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combina transformaciones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])




# Variables predictoras (X) y variable objetivo (y)
X = df[['Age', 'JobSatisfaction', 'Education', ...]]  # Incluye las variables relevantes
y = df['Attrition']

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)







# Combina preprocesamiento con el modelo
model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=42))])

# Entrena el modelo
model.fit(X_train, y_train)


# Combina preprocesamiento con el modelo
model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=42))])

# Entrena el modelo
model.fit(X_train, y_train)







# Predice en el conjunto de prueba
y_pred = model.predict(X_test)

# Evalúa el rendimiento
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))