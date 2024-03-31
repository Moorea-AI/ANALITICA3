################################################################
#               SEGUNDO TRABAJO PARA LA MATERIA:               #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE MARKETING                         #
#                              POR:                            #
#                                                              #
#                    AURA LUZ MORENO - MOOREA                  #
#                       ALEJANDRA AGUIRRE                      #
#                                                              #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################



import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go
import plotly.express as px
import a_funciones as fn




# Conectarse a la base de datos
conn = sql.connect('db_movies')

# Crear el cursor
cur = conn.cursor()

# Para ver las tablas
cur.execute("SELECT name FROM sqlite_master where type='table' ")
cur.fetchall()

# Traer tablas de BD a Python

movies = pd.read_sql("""SELECT *  FROM movies""", conn)
movie_ratings = pd.read_sql('SELECT * FROM ratings', conn)

# Identificar campos de cruce y verificar que estén en mismo formato ####
# verificar duplicados

movies.info()
movies.head()
movies.duplicated().sum()

movie_ratings.info()
movie_ratings.head()
movie_ratings.duplicated().sum()

# calcular la distribución de calificaciones
cr = pd.read_sql(""" 
    SELECT 
        "Rating" AS rating, 
        COUNT(*) AS conteo 
    FROM ratings 
    GROUP BY "Rating" 
    ORDER BY conteo DESC
""", conn)

print(cr)


fig = go.Figure(data=[go.Bar(x=cr['rating'], y=cr['conteo'], text=cr['conteo'], textposition="outside")])

# Actualizar el diseño
fig.update_layout(title="Conteo de calificaciones", xaxis={'title': 'Calificación'}, yaxis={'title': 'Conteo'})

# Mostrar la figura
fig.show()

# calcular cada usuario cuántas peliculas calificó
rating_users=pd.read_sql(''' select "userId" as user_id,
                         count(*) as cnt_rat
                         from ratings
                         group by "userId"
                         order by cnt_rat asc
                         ''',conn)
print(rating_users)

fig  = px.histogram(rating_users, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones por usario')
fig.show()

# excluir usuarios con menos de 50 peliculas calificadas (para tener calificaion confiable) y los que tienen mas de mil porque pueden ser no razonables

rating_users2=pd.read_sql(''' select "userId" as user_id,
                         count(*) as cnt_rat
                         from ratings
                         group by "userId"
                         having cnt_rat >=50 and cnt_rat <=1000
                         order by cnt_rat asc
                         ''',conn )

# ver distribucion despues de filtros,ahora se ve mas razonables
print(rating_users2.describe())


# graficar distribucion despues de filtrar datos
fig  = px.histogram(rating_users2, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones por usario')
fig.show()

# verificar cuantas calificaciones tiene cada pelicula
rating_movies =pd.read_sql(''' select movieId ,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         order by cnt_rat desc
                         ''',conn )
print(rating_movies)
print(rating_movies.describe())
fig  = px.histogram(rating_movies, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones para cada pelicula')
fig.show()
# verificar cuantas calificaciones tiene cada pelicula, filtrando entre 20 y 100
rating_movies1 =pd.read_sql(''' select movieId ,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         having cnt_rat >=20 and cnt_rat <=150
                         order by cnt_rat desc
                         ''',conn )
print(rating_movies1)
print(rating_movies1.describe())
fig  = px.histogram(rating_movies1, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones para cada pelicula')
fig.show()
#
fn.ejecutar_sql('preprocesamientos.sql', cur)

cur.execute("select name from sqlite_master where type='table' ")
cur.fetchall()


# verficar tamaño de tablas con filtros

# movies

pd.read_sql('select count(*) from movies', conn)
pd.read_sql('select count(*) from movies_final', conn)

# ratings
pd.read_sql('select count(*) from ratings', conn)
pd.read_sql('select count(*) from ratings_final', conn)

# 3 tablas cruzadas ###
pd.read_sql('select count(*) from full_ratings', conn)

ratings=pd.read_sql('select * from full_ratings',conn)
print(ratings.duplicated().sum()) # al cruzar tablas a veces se duplican registros
print(ratings.info())
print(ratings.head(10))


