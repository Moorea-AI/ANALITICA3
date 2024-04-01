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


# Hacemos la conexión a la base de datos db_movies
conn = sql.connect('data/db_movies')
cur = conn.cursor()

#Revisamos las tablas que contiene la BD
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()


for table in tables:
    print(table[0])

# Tablas de la base de datos 
# ratings
# movies
# usuarios_sel
# movies_sel
# ratings_final
# users_final
# movies_final

# Exploración de cada tabla de la Base de datos:

# Tabla ratings
df_ratings = pd.read_sql_query("SELECT * FROM ratings LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'ratings':")
print(df_ratings)

#Tipos de datos de la tabla ratings
cur.execute("PRAGMA table_info(ratings)")
columns_ratings = cur.fetchall()
print("Tipos de datos de las columnas en la tabla 'ratings':")
for column in columns_ratings:
    print(column[1] + ": " + column[2])
    
    

# Tabla movies
df_movies = pd.read_sql_query("SELECT * FROM movies LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'movies':")
print(df_movies)

#Tipos de datos de la tabla movies
cur.execute("PRAGMA table_info(movies)")
columns_movies = cur.fetchall()
print("\nTipos de datos de las columnas en la tabla 'movies':")
for column in columns_movies:
    print(column[1] + ": " + column[2])




# Tabla usuarios_sel
df_usuarios_sel = pd.read_sql_query("SELECT * FROM usuarios_sel LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'usuarios_sel':")
print(df_usuarios_sel)

# Tabla movies_sel
df_movies_sel = pd.read_sql_query("SELECT * FROM movies_sel LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'movies_sel':")
print(df_movies_sel)

# Tabla ratings_final
df_ratings_final = pd.read_sql_query("SELECT * FROM ratings_final LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'ratings_final':")
print(df_ratings_final)

# Tabla users_final
df_users_final = pd.read_sql_query("SELECT * FROM users_final LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'users_final':")
print(df_users_final)

# Tabla movies_final
df_movies_final = pd.read_sql_query("SELECT * FROM movies_final LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'movies_final':")
print(df_movies_final)



# Traemos las tablas a Python
df_movies = pd.read_sql("""SELECT *  FROM movies""", conn)
df_ratings = pd.read_sql('SELECT * FROM ratings', conn)

# La tabla df_movies tiene 9742 filas y 3 columnas: movieId, title y genre
# La tabla df_ratings tiene 100836 filas y 4 columnas: " ", userId, movieId, rating, timestamp

# Verificamos la información de la tabla y sus datos duplicados: NO TIENE y sus datos nulos: NO TIENE
df_movies.info()
df_movies.head()
df_movies.duplicated().sum()

# Verificamos la información de la tabla y sus datos duplicados: NO TIENE y sus datos nulos: NO TIENE
df_ratings.info()
df_ratings.head()
df_ratings.duplicated().sum()


# Ahora empezaremos a hacernos preguntas. Cuál es la calificación más frecuente? 
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

#    rating  conteo
# 0     4.0   26818
# 1     3.0   20047
# 2     5.0   13211
# 3     3.5   13136
# 4     4.5    8551
# 5     2.0    7551
# 6     2.5    5550
# 7     1.0    2811
# 8     1.5    1791
# 9     0.5    1370



calificaciones_por_usuario_query = """
    SELECT "userId" AS user_id,
           COUNT(*) AS cnt_rat
    FROM ratings
    GROUP BY "userId"
    ORDER BY cnt_rat DESC
"""

calificaciones_por_usuario = pd.read_sql_query(calificaciones_por_usuario_query, conn)
fig = px.bar(calificaciones_por_usuario, x='user_id', y='cnt_rat', text='cnt_rat')
fig.update_layout(title='Número de Calificaciones por Usuario',
                  xaxis_title='ID de Usuario',
                  yaxis_title='Número de Calificaciones')
fig.show()

# Podemos observar que hay usuarios que han sobrepasado las 2500 calificaciones
# Pero limitaremos esto, ya que esto puede sesgar los datos.
rating_users2=pd.read_sql(''' select "userId" as user_id,
                         count(*) as cnt_rat
                         from ratings
                         group by "userId"
                         having cnt_rat >=50 and cnt_rat <=1000
                         order by cnt_rat asc
                         ''',conn )

print(rating_users2.describe())
fig  = px.histogram(rating_users2, x= 'cnt_rat', title= 'Frecuencia de calificaciones por usuario')
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


