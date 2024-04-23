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


# Se hace llamado al archivo de funciones
import a_funciones as fn # Archivo de funciones definidas por mi

data = pd.read_csv("data/master.csv")
data.head()
