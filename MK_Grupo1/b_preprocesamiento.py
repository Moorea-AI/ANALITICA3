################################################################
#               SEGUNDO TRABAJO PARA LA MATERIA:               #
#                 APLICACIONES DE LA ANALITICA                 #
#                     MÓDULO DE MARKETING                      #
#                              POR:                            #
#                                                              #
#                    AURA LUZ MORENO - MOOREA                  #
#                       ALEJANDRA AGUIRRE                      #
#                                                              #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################

import sqlite3 as sql 
import pandas as pd
import a_funciones as a_funciones 
import plotly.graph_objs as go ### para gráficos
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder 

# Conectarse a la base de datos
conn = sql.connect('data/db_movies')

# Crear el cursor
cur = conn.cursor()

# Para ver las tablas
cur.execute("SELECT name FROM sqlite_master where type='table' ")
cur.fetchall() 

############ Traer tablas de BD a Python ############

movies= pd.read_sql("""SELECT *  FROM movies""", conn)
movie_ratings = pd.read_sql('SELECT * FROM ratings', conn)
