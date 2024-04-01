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


#from mlxtend.preprocessing import TransactionEncoder


#Conexión a la BD
conn = sql.connect('data\db_movies')
cur = conn.cursor()

#Revisamos las tablas que contiene la BD
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()


for table in tables:
    print(table[0])


df_movies = pd.read_sql_query("SELECT * FROM movies LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'movies':")
print(df_movies)

cur.execute("PRAGMA table_info(ratings)")
columns_ratings = cur.fetchall()

print("Tipos de datos de las columnas en la tabla 'ratings':")
for column in columns_ratings:
    print(column[1] + ": " + column[2])


cur.execute("PRAGMA table_info(movies)")
columns_movies = cur.fetchall()


print("\nTipos de datos de las columnas en la tabla 'movies':")
for column in columns_movies:
    print(column[1] + ": " + column[2])
    
    
query = f"SELECT * FROM {'ratings'}"
df_ratings = pd.read_sql_query(query, conn)


    
    
df_ratings = pd.read_sql_query("SELECT * FROM ratings LIMIT 5;", conn)
print("Primeras filas de la tabla 'ratings':")
print(df_ratings)


movies = pd.read_sql("""SELECT *  FROM movies""", conn)
movie_ratings = pd.read_sql('SELECT * FROM ratings', conn)

consulta_sql = """
    SELECT *
    FROM ratings
"""
ratings = pd.read_sql(consulta_sql, conn)


consulta_sql = """
    SELECT *
    FROM movies
"""
movies = pd.read_sql(consulta_sql, conn)



movies.info()
movies.head()
movies.duplicated().sum()

movie_ratings.info()
movie_ratings.head()
movie_ratings.duplicated().sum()

# calcular la distribución de calificaciones
Calificaciones = pd.read_sql(""" 
    SELECT 
        "Rating" AS rating, 
        COUNT(*) AS conteo 
    FROM ratings 
    GROUP BY "Rating" 
    ORDER BY conteo DESC
""", conn)

print(Calificaciones)


fig = go.Figure(data=[go.Bar(x=Calificaciones['rating'], y=Calificaciones['conteo'], text=Calificaciones['conteo'], textposition="outside")])
fig.update_layout(title="Conteo de calificaciones", xaxis={'title': 'Calificación'}, yaxis={'title': 'Conteo'})
fig.show()

rating_users=pd.read_sql(''' select "userId" as user_id,
                         count(*) as cnt_rat
                         from ratings
                         group by "userId"
                         order by cnt_rat asc
                         ''',conn)
print(rating_users)

fig  = px.histogram(rating_users, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones por usario')
fig.show()





rating_users2=pd.read_sql(''' select "userId" as user_id,
                         count(*) as cnt_rat
                         from ratings
                         group by "userId"
                         having cnt_rat >=50 and cnt_rat <=1000
                         order by cnt_rat asc
                         ''',conn )

print(rating_users2.describe())


fig  = px.histogram(rating_users2, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones por usario')
fig.show()




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


fn.ejecutar_sql('preprocesamiento1.sql', cur)

cur.execute("select name from sqlite_master where type='table' ")
cur.fetchall()


pd.read_sql('select count(*) from movies', conn)
pd.read_sql('select count(*) from movies_final', conn)
pd.read_sql('select count(*) from ratings', conn)
pd.read_sql('select count(*) from ratings_final', conn)
pd.read_sql('select count(*) from full_ratings', conn)

ratings=pd.read_sql('select * from full_ratings',conn)
print(ratings.duplicated().sum()) # al cruzar tablas a veces se duplican registros
print(ratings.info())
print(ratings.head(10))

movies=pd.read_sql("""select * from movies""", conn)
genres=movies['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)


