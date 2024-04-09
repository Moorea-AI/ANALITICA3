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

import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder

# Se hace llamado al archivo de funciones
import a_funciones as fn


# Hacemos la conexión a la base de datos db_movies
conn = sql.connect('data/db_movies')
cur = conn.cursor()

# Revisamos las tablas que contiene la BD
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()


for table in tables:
    print(table[0])

# Tablas de la base de datos:
# ratings
# movies
# usuarios_sel
# movies_sel
# ratings_final
# users_final
# movies_final

#### Exploración de cada tabla de la Base de datos:

# Tabla ratings
df_ratings = pd.read_sql_query("SELECT * FROM ratings LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'ratings':")
print(df_ratings)


# Primeras filas de la tabla 'ratings':
#    userId  movieId  rating  timestamp
# 0       1        1     4.0  964982703
# 1       1        3     4.0  964981247
# 2       1        6     4.0  964982224
# 3       1       47     5.0  964983815
# 4       1       50     5.0  964982931


#Tipos de datos de la tabla ratings
cur.execute("PRAGMA table_info(ratings)")
columns_ratings = cur.fetchall()
print("Tipos de datos de las columnas en la tabla 'ratings':")
for column in columns_ratings:
    print(column[1] + ": " + column[2])
    
# Tipos de datos de las columnas en la tabla 'ratings':
# userId: INTEGER
# movieId: INTEGER
# rating: REAL
# timestamp: INTEGER    

# Tabla movies
df_movies = pd.read_sql_query("SELECT * FROM movies LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'movies':")
print(df_movies)


# Primeras filas de la tabla 'movies':
#    movieId                               title  \
# 0        1                    Toy Story (1995)   
# 1        2                      Jumanji (1995)   
# 2        3             Grumpier Old Men (1995)   
# 3        4            Waiting to Exhale (1995)   
# 4        5  Father of the Bride Part II (1995)   

#                                         genres  
# 0  Adventure|Animation|Children|Comedy|Fantasy  
# 1                   Adventure|Children|Fantasy  
# 2                               Comedy|Romance  
# 3                         Comedy|Drama|Romance  
# 4                                       Comedy 


#Tipos de datos de la tabla movies
cur.execute("PRAGMA table_info(movies)")
columns_movies = cur.fetchall()
print("\nTipos de datos de las columnas en la tabla 'movies':")
for column in columns_movies:
    print(column[1] + ": " + column[2])


# Tipos de datos de las columnas en la tabla 'movies':
# movieId: INTEGER
# title: TEXT
# genres: TEXT


# Tabla usuarios_sel
df_usuarios_sel = pd.read_sql_query("SELECT * FROM usuarios_sel LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'usuarios_sel':")
print(df_usuarios_sel)

# Primeras filas de la tabla 'usuarios_sel':
#    user_id  cnt_rat
# 0      182      977
# 1      307      975
# 2      603      943
# 3      298      939
# 4      177      904


# Tabla movies_sel
df_movies_sel = pd.read_sql_query("SELECT * FROM movies_sel LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'movies_sel':")
print(df_movies_sel)


# Primeras filas de la tabla 'movies_sel':
#    movieId  cnt_rat
# 0    58559      149
# 1     6539      149
# 2     1214      146
# 3      595      146
# 4     1036      145




# Tabla ratings_final
df_ratings_final = pd.read_sql_query("SELECT * FROM ratings_final LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'ratings_final':")
print(df_ratings_final)



# Primeras filas de la tabla 'ratings_final':
#    user_id  movie_rating  movieId
# 0        1           4.0        3
# 1        1           4.0        6
# 2        1           3.0       70
# 3        1           5.0      101
# 4        1           5.0      151



# Tabla users_final
df_users_final = pd.read_sql_query("SELECT * FROM users_final LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'users_final':")
print(df_users_final)

# Primeras filas de la tabla 'users_final':
#    movieId  userId  rating
# 0        1       1     4.0
# 1        3       1     4.0
# 2        6       1     4.0
# 3       47       1     5.0
# 4       50       1     5.0

# Tabla movies_final
df_movies_final = pd.read_sql_query("SELECT * FROM movies_final LIMIT 5;", conn)
print("\nPrimeras filas de la tabla 'movies_final':")
print(df_movies_final)



# Primeras filas de la tabla 'movies_final':
#    movieId                               title                      genres
# 0        2                      Jumanji (1995)  Adventure|Children|Fantasy
# 1        3             Grumpier Old Men (1995)              Comedy|Romance
# 2        5  Father of the Bride Part II (1995)                      Comedy
# 3        6                         Heat (1995)       Action|Crime|Thriller
# 4        7                      Sabrina (1995)  


# Traemos las tablas a Python
df_movies = pd.read_sql("""SELECT *  FROM movies""", conn)
df_ratings = pd.read_sql('SELECT * FROM ratings', conn)

# La tabla df_movies tiene 9742 filas y 3 columnas: movieId, title y genre
# La tabla df_ratings tiene 100836 filas y 4 columnas: " ", userId, movieId, rating, timestamp

# Verificamos la información de la tabla y sus datos duplicados: NO TIENE y sus datos nulos: NO TIENE
nulosmovies = df_movies.info()   # NO TIENE NULOS
df_movies.head()
df_movies.duplicated().sum()  #NO TIENE DUPLICADOS
duplicadosmovies = df_movies.duplicated().sum()

print("\nnulos de 'movies':")
print(nulosmovies)

print("\nduplicados de 'movies':")
print(duplicadosmovies)


# Verificamos la información de la tabla y sus datos duplicados: NO TIENE y sus datos nulos: NO TIENE
nulosratings = df_ratings.info()   #NO TIENE NULOS
df_ratings.head()   #Observamos que el timestamp esta raro, tendremos que ver más adelante si se necesita o no, y que significa ese dato numérico
duplicadosratings =  df_ratings.duplicated().sum()  #NO TIENE DUPLICADOS

print("\nnulos de 'ratings':")
print(nulosratings)

print("\nduplicados de 'ratings':")
print(duplicadosratings)

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

print(Calificaciones)  #Podemos observar que 3,4,5 son las calificaciones más frecuentes

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



# Cuántas veces ha sido calificada una película?
# Podemos ver que la película 356 ha sido calificada 329 veces

rating_movies =pd.read_sql(''' select movieId ,
                         count(*) as cnt_rat
                         from ratings
                         group by "movieId"
                         order by cnt_rat desc
                         ''',conn )
print(rating_movies)
print(rating_movies.describe())
fig  = px.histogram(rating_movies, x= 'cnt_rat', title= 'Frecuencia de calificaciones para cada pelicula')
fig.show()



# Seleccionamos las que tienen entre 20 y 150 calificaciones
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


# Cuáles usuarios han calificados más de 100 peliculas?
# Esto nos lleva a la pregunta: Es lógico que un usuario haya calificado 2.698 peliculas
# Tenemos datos atipicos? errores en la base de datos?
consulta_sql = """
    SELECT userId, COUNT(DISTINCT movieId) AS numero_peliculas
    FROM ratings
    GROUP BY userId
    HAVING numero_peliculas >= 100
    order by numero_peliculas desc
"""
pd.read_sql(consulta_sql, conn)



# Cuál es la pelicula con la calificación más alta?
consulta_sql = """
    SELECT m.title, AVG(r.rating) as calificacion
    FROM movies m
    LEFT JOIN ratings r ON m.movieId = r.movieId
    GROUP BY m.title
    ORDER BY calificacion DESC 
    LIMIT 1
"""
pd.read_sql(consulta_sql, conn)

#0	Zeitgeist: Moving Forward (2011)	5.0


# Cuáles son las 5 peliculas más recomendadas?
consulta_sql = """
    SELECT m.title, COUNT(r.rating) as total_calificacion
    FROM movies m
    LEFT JOIN ratings r ON m.movieId = r.movieId
    GROUP BY m.title
    ORDER BY total_calificacion DESC 
    LIMIT 5
"""
pd.read_sql(consulta_sql, conn)

# title	total_calificacion
# 0	Forrest Gump (1994)	329
# 1	Shawshank Redemption, The (1994)	317
# 2	Pulp Fiction (1994)	307
# 3	Silence of the Lambs, The (1991)	279
# 4	Matrix, The (1999)	278


# Distribución de las calificaciones
consulta_sql = """
    SELECT rating, COUNT(*) AS n_movies
    FROM ratings
    GROUP BY rating
    ORDER BY n_movies DESC
"""

cr=pd.read_sql(consulta_sql, conn)
data  = go.Bar( x=cr.rating,y=cr.n_movies, text=cr.n_movies, textposition="outside")
Layout=go.Layout(title="Count of ratings",xaxis={'title':'Rating'},yaxis={'title':'Count movies'})
go.Figure(data,Layout)


query = f"SELECT * FROM {'ratings'}"
df_ratings = pd.read_sql_query(query, conn)

query = f"SELECT * FROM {'movies'}"
df_movies = pd.read_sql_query(query, conn)

# Ahora bien, crearemos una sola base de datos para poder trabajar con esta. Lo filtrremos por el movieId que es el campo comun en ambos
# Pero antes, analizaremos detalladamente ambas bases de datos

print("\nCruce de datos:")
print("Tabla ratings:", df_ratings.columns)
print("Tabla movies:", df_movies.columns)

# Cruce de datos:
# Tabla ratings: Index(['userId', 'movieId', 'rating', 'timestamp'], dtype='object')
# Tabla movies: Index(['movieId', 'title', 'genres'], dtype='object')


#Verificamos y podemos ver que ambos son tipo Int
print("\nFormato de 'movieId':")
print("Tabla ratings:", df_ratings['movieId'].dtype)
print("Tabla movies:", df_movies['movieId'].dtype)

# Cruce de datos:
# Tabla ratings: Index(['userId', 'movieId', 'rating', 'timestamp'], dtype='object')
# Tabla movies: Index(['movieId', 'title', 'genres'], dtype='object')


# Ahora bien, verifiquemos si son campos unicos o tienen duplicados. Podemos observar que NO existen datos duplicados (ya lo habiamos mostrado más arriba)
duplicadosratings = df_ratings.duplicated().sum()
duplicadosmovies = df_movies.duplicated().sum()


# Calcular cuántas películas calificó cada usuario
rating_users = pd.read_sql('''SELECT userId as user_id,
                                     COUNT(*) as cnt_rat
                              FROM ratings
                              GROUP BY userId
                              ORDER BY cnt_rat DESC''', conn)

# Imprimir el DataFrame
print(rating_users)


rating_users.describe()

# 	user_id	cnt_rat
# count	610.000000	610.000000
# mean	305.500000	165.304918
# std	176.236111	269.480584
# min	1.000000	20.000000
# 25%	153.250000	35.000000
# 50%	305.500000	70.500000
# 75%	457.750000	168.000000
# max	610.000000	2698.000000

# Con esto tenemos que los usuarios calificaron un promedio de 165 peliculas
# Los usuarios que menos han calificado son 20 peliculas y los que más han calificado son 2698 peliculas
# Existirán usuarios que hayan tenido el tiempo suficiente para calificar 2698 peliculas?

# El 25% de los usuarios han calificado en promedio 35 peliculas
# el 50% de los usuarios han calificado alrededor 70 peliculas

# Por ende, tomaremos los usuarios que han calificado entre 30 y 1000.

# Consulta SQL para filtrar usuarios con más de 50 calificaciones pero menos de 1000
query = '''
    SELECT userId as user_id,
           COUNT(*) as cnt_rat
    FROM ratings
    GROUP BY userId
    HAVING cnt_rat >= 30 AND cnt_rat <= 1000
    ORDER BY cnt_rat DESC
'''


# Ejecutar la consulta y cargar los resultados en un DataFrame
rating_users_def = pd.read_sql(query, conn)

# Imprimir el DataFrame
print(rating_users_def)

# Esto quedaría gráficamente asi:

fig  = px.histogram(rating_users_def, x= 'cnt_rat', title= 'Frecuencia de calificaciones por usuario')
fig.show()
     
     
rating_users_def.describe()

# Con esto, los datos atipicos quedan más controlados
# user_id	cnt_rat
# count	489.000000	489.000000
# mean	299.171779	163.014315
# std	176.961279	177.733228
# min	1.000000	30.000000
# 25%	142.000000	50.000000
# 50%	302.000000	94.000000
# 75%	450.000000	196.000000
# max	609.000000	977.000000

# Ejecutamos el archivo preprocesamiento.sql para limpiar la base de datos
fn.ejecutar_sql('preprocesamiento.sql', cur)

cur.execute("select name from sqlite_master where type='table' ")
cur.fetchall()

final = pd.read_sql("""SELECT *  FROM final""", conn)
final

# Ahora bien, debemos organizar los géneros de las peliculas para que sean medibles
genres=final['genres'].str.split('|') 
te = TransactionEncoder() 
genres = te.fit_transform(genres) 
genres = pd.DataFrame(genres, columns = te. columns_) 


final = pd.concat([final, genres], axis=1)
final = final.drop('genres', axis=1)
final

final[final['(no genres listed)']]


# Extraemos el año de la columna title en otra columna 
final['year'] = final['title'].str.extract('.*\((.*)\).*', expand=True)
final
# Borramos el texto de los paréntesis de la columna title
final['title'] = final['title'].str.replace(r'\s*\([^()]*\)', '', regex=True)
final

# Reemplazamos en los géneros el false por 0 y el true por 1
final[genres.columns] = final[genres.columns].replace({False: 0, True: 1})
final

# Guardamos la base final en sql
final.to_sql('final', conn, if_exists='replace', index=False)