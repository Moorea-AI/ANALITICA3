################################################################
#               PRIMER TRABAJO PARA LA MATERIA:                #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE RECURSOS HUMANOS                  #
#                              POR:                            #
#                       ALEJANDRA AGUIRRE                      #
#                    AURA LUZ MORENO - MOOREA                  #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################


### Este es el archivo de funciones donde se recopilan todas las que se usarán y serán llamadas 
# desde otros archivos. Usaremos una matriz de confusión y una de correlación para ver la cohesión de 
#las variables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Para poder ver gráficamente las variables
import seaborn as sns #Para poder ver gráficamente las variables.
import os  #Paquete OS: https://docs.python.org/es/3.10/library/os.html
#Ya que es un problema de clasificación, quiero usar la matriz de confusion para 
# visualizar mejor la relación de las variables que influyen si un empleado se queda o no.
from sklearn.metrics import confusion_matrix 
#Se toman del archivo del profe por si nos sirven de algo más adelante:
from sklearn.impute import SimpleImputer 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 


#Esta función permite ejecutar un archivo  con extensión .sql que contenga varias consultas
#En el archivo de SQL pondremos todas las consultas necesarias para construir la base de datos del modelo
def ejecutar_sql(nombre_archivo, cur):
    try:
        # Mirar si el archivo existe
        if not os.path.exists(nombre_archivo):
            raise FileNotFoundError(f"El archivo {nombre_archivo} no se encuentra")
        
        # Abir el archivo SQL
        with open(nombre_archivo, 'r') as sql_file:
            sql_as_string = sql_file.read()
        
        # Ejecutar el SQL
        cur.executescript(sql_as_string)
        
    except Exception as e:
        # Imprimir el error si lo hay
        print("Error al ejecutar el archivo SQL:", str(e))


# esta podemos usarla para limpiar las bases de datos, del semestre pasado
def identify_and_remove_outliers(conn, columns, threshold=2.1):
    # Leer datos desde la base de datos SQL
    df = pd.read_sql("SELECT * FROM all_employees", conn)

    for column in columns:
        Q1 = np.quantile(df[column], 0.25)
        Q3 = np.quantile(df[column], 0.75)
        IQR = Q3 - Q1
        upper = Q3 + threshold * IQR
        lower = Q1 - threshold * IQR

        # Usar SQL para eliminar outliers
        query = f"DELETE FROM all_employees WHERE {column} > {upper} OR {column} < {lower}"
        conn.execute(query)
        conn.commit()
        
        
#Se podria hacer una función para determinar la matriz de confusion
#Este lo tomo del trabajo del semestre pasado para ver si funciona.

def show_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return matrix


#IMPUTADORES:

def imputar_f (df,list_cat):  
        
    
    df_c=df[list_cat]
    df_n=df.loc[:,~df.columns.isin(list_cat)]

    imputer_n=SimpleImputer(strategy='median')
    imputer_c=SimpleImputer(strategy='most_frequent')

    imputer_n.fit(df_n)
    imputer_c.fit(df_c)

    X_n=imputer_n.transform(df_n)
    X_c=imputer_c.transform(df_c)

    df_n=pd.DataFrame(X_n,columns=df_n.columns)
    df_c=pd.DataFrame(X_c,columns=df_c.columns)

    df =pd.concat([df_n,df_c],axis=1)
    return df


def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac


def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["reg_lineal","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos



def preparar_datos (df):
   #######Cargar y procesar nuevos datos ######
   #### Cargar modelo y listas 
   
    list_cat=joblib.load("list_cat.pkl")
    list_dummies=joblib.load("list_dummies.pkl")
    var_names=joblib.load("var_names.pkl")
    scaler=joblib.load( "scaler.pkl") 

    ####Ejecutar funciones de transformaciones
    
    df=imputar_f(df,list_cat)
    df_dummies=pd.get_dummies(df,columns=list_dummies)
    df_dummies= df_dummies.loc[:,~df_dummies.columns.isin(['perf_2023','EmpID2'])]
    X2=scaler.transform(df_dummies)
    X=pd.DataFrame(X2,columns=df_dummies.columns)
    X=X[var_names]
     
    return X
