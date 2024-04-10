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

import numpy as np
import pandas as pd
import sqlite3 as sql
import a_funciones as fn ## para procesamiento
import openpyxl
from mlxtend.preprocessing import TransactionEncoder

####Paquete para sistema basado en contenido ####
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors

conn=sql.connect('data/db_movies_2')
cur=conn.cursor()

fn.ejecutar_sql('preprocesamiento.sql', cur)

movies=pd.read_sql('select * from df_movies', conn )
ratings=pd.read_sql('select * from df_ratings', conn)