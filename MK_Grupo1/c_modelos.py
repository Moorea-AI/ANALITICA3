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
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib
from mlxtend.preprocessing import TransactionEncoder
import a_funciones as fn
import plotly.graph_objs as go 
import plotly.express as px



#Para activar el paquete de Surprise se debe cambiar el kernel por el de conda
# Una vez se abre el powershell, conda install -c conda-forge scikit-surprise
from surprise import Reader, Dataset
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import cross_validate, GridSearchCV


#### conectar_base_de_Datos

conn=sql.connect('data//db_movies')
cur=conn.cursor()

#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

cur.execute("PRAGMA table_info('df_final')")
columns = cur.fetchall()
for column in columns:
    print(column)


# #Exploremos las tablas finales de la BD.
# [('ratings',),            Original
#  ('movies',),             Original
#  ('users_final',),        Nueva

#  ('usuarios_sel',),       Nueva
#  ('movies_sel',),         Nueva

#  ('movies_final',),       Nueva
#  ('ratings_final',),      Nueva

#  ('df_final',),           Nueva después del preprocesamiento
#  ('df_final2',),          Nueva
#  ('recommendations',),    Nueva con el resultado del ejercicio

# 1. sistemas basados en popularidad 
##### recomendaciones basado en popularidad ######

#¿Cuáles son las 10 películas con la calificación promedio más alta?
consulta_sql = """
    SELECT title, AVG(rating) as calificacion
    FROM df_final
    GROUP BY title
    ORDER BY calificacion DESC 
    LIMIT 10
"""
pd.read_sql(consulta_sql, conn)

#### ¿Cuáles son las 10 películas más populares además del total de calificaciones que han recibido?
consulta_sql = """
    SELECT title, COUNT(rating) as total_calificacion
    FROM df_final
    GROUP BY title
    ORDER BY total_calificacion DESC 
    LIMIT 10
"""
pd.read_sql(consulta_sql, conn)





# Año de las peliculas más populares
sql1 = pd.read_sql('''  SELECT year, count(title) AS numberOfMovies 
                    FROM final 
                    GROUP BY year 
                    ORDER BY numberOfMovies DESC
                            ''',conn )
# Graficamos el año de las peliculas más populares
fig = px.bar(sql1, x='year', y='numberOfMovies', title='Cantidad de peliculas por año')
fig.show()



cursor.execute('SELECT * FROM df_final')
columns = cursor.fetchall()
for column in columns:
    print(column)
    


# Consulta para obtener películas con 6 o 7 géneros
sql2 = pd.read_sql('''  SELECT title, (Action + Adventure + Animation + Children + Comedy + Crime + Documentary + 
                            Drama + Fantasy + 'Film-Noir' + Horror + IMAX + Musical + Mystery + 
                            Romance + 'Sci-Fi' + Thriller + War + Western) AS total_genres 
                        FROM df_final
                        GROUP BY title
                        HAVING total_genres IN (6, 7)
                        ORDER BY total_genres desc''',conn )


# Visualización de películas por cantidad de géneros
fig = px.bar(sql2, x='title', y='total_genres', title='Cantidad de géneros por pelicula')
fig.show()


# Consulta para obtener la cantidad de películas por género
sql3 = pd.read_sql('''   SELECT 
    SUM(Action) AS Action, 
    SUM(Adventure) AS Adventure, 
    SUM(Animation) AS Animation,
    SUM(Children) AS Children,
    SUM(Comedy) AS Comedy,
    SUM(Crime) AS Crime,
    SUM(Documentary) AS Documentary,
    SUM(Drama) AS Drama,
    SUM(Fantasy) AS Fantasy,
    SUM("Film-Noir") AS "Film-Noir",
    SUM(Horror) AS Horror,
    SUM(IMAX) AS IMAX,
    SUM(Musical) AS Musical,
    SUM(Mystery) AS Mystery,
    SUM(Romance) AS Romance,
    SUM("Sci-Fi") AS "Sci-Fi",
    SUM(Thriller) AS Thriller,
    SUM(War) AS War,
    SUM(Western) AS Western 
    FROM df_final; ''',conn )


# Gráfico de barras con los géneros más populares
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
counts = [sql3[genre][0] for genre in genres]

fig = go.Figure(data=[go.Bar(x=genres, y=counts)])
fig.update_layout(title='Géneros más populares')
fig.show()

# Consulta para obtener películas con una calificación promedio mayor o igual a 4.5
sql4 = pd.read_sql(''' SELECT movieId, title, AVG(rating) AS average_rating
FROM df_final
GROUP BY movieId, title
HAVING AVG(rating) >= 4.5
ORDER BY average_rating DESC;
''',conn )

# Visualización de estas películas con un gráfico de barras
fig = go.Figure(data=[go.Bar(x=sql4['title'], y=sql4['average_rating'])])
fig.update_layout(title='Calificación promedio por peli >= 4.5')
fig.show()

# Consulta para obtener las 10 películas mejor calificadas con al menos 5 vistas
sql5 = pd.read_sql("""SELECT title,
            AVG(rating) AS avg_rat,
            COUNT(*) AS view_num
            FROM df_final
            GROUP BY title
            HAVING view_num >= 5
            ORDER BY avg_rat DESC
            LIMIT 10
            """, conn)

# Gráfico referente a la consulta anterior
fig = go.Figure(data=[go.Bar(x=sql5['title'], y=sql5['avg_rat'])])
fig.update_layout(title='Top 10 Películas con Mejor Calificación Promedio (con al menos 5 vistas)',
                    xaxis_title='Película',
                    yaxis_title='Calificación Promedio')
fig.show()

# Consulta para obtener la mejor película calificada por género
sql6 = pd.read_sql(""" SELECT genero, title, MAX(rating) AS mejor_calificacion
FROM (
    SELECT title,
            CASE
                WHEN Action = 1 THEN 'Accion'
                WHEN Adventure = 1 THEN 'Aventura'
                WHEN Animation = 1 THEN 'Animacion'
                WHEN Children = 1 THEN 'Infantil'
                WHEN Comedy = 1 THEN 'Comedia'
                WHEN Crime = 1 THEN 'Crimen'
                WHEN Documentary = 1 THEN 'Documental'
                WHEN Drama = 1 THEN 'Drama'
                WHEN Fantasy = 1 THEN 'Fantasia'
                WHEN Horror = 1 THEN 'Horror'
                WHEN IMAX = 1 THEN 'Imax'
                WHEN Musical = 1 THEN 'Musical'
                WHEN Mystery = 1 THEN 'Misterio'
                WHEN Romance = 1 THEN 'Romance'
                WHEN Thriller = 1 THEN 'Thrill'
                WHEN War = 1 THEN 'Guerra'
                WHEN Western = 1 THEN 'Occidental'
                WHEN "Sci-Fi" = 1 THEN 'Ciencia ficcion'
                WHEN "Film-Noir" = 1 THEN 'Cine negro'
            END as genero,
            rating
    FROM df_final
) AS generos
GROUP BY genero;
""", conn)

# Visualización de estas películas por género
fig = px.bar(sql6, x='genero', y='mejor_calificacion', color='genero', title='Mejor calificación por género', text='title')
fig.update_layout(showlegend=False)
fig.show()

# Consulta para obtener las películas mejor calificadas por año de lanzamiento
sql7 = pd.read_sql('''SELECT title, year, AVG(rating) AS average_rating
                    FROM df_final
                    GROUP BY year, title
                    ORDER BY year, average_rating DESC''', conn)

fig = px.bar(sql7, x='year', y='average_rating', color='title', title='Peliculas mejor calificas por año de lanzamiento')
fig.update_layout(
    autosize=False,
    width=1000,
    height=600,
    margin=dict(l=50, r=50, b=100, t=100, pad=4)
)
fig.show()
















# 2.1 Sistema de recomendación basado en contenido un solo producto - Manual 


# Cargar todos los registros de la tabla 'final' en un DataFrame
final = pd.read_sql_query('SELECT * FROM final', conn) # Carga la tabla 'final' en el DataFrame 'final'


# Selecciona solo las columnas relevantes (elimina las columnas 'userId' y 'movieId')
general = final.loc[:,~final.columns.isin(['userId','movieId'])]
general

# Guarda el DataFrame 'general' en la base de datos con el nombre 'final2'
general.to_sql('final2', conn, if_exists='replace', index=False)

# Consulta SQL para agrupar por título y año y calcular la calificación promedio
final1 = pd.read_sql("""select *, avg(rating) as avg_rat
                    from final2
                    group by year, title
                    order by year desc, avg_rat desc""", conn) # Agrupa y ordena por año y calificación promedio
final1

# Convierte la columna 'year' a tipo entero y escala sus valores entre 0 y 1
final1['year']=final1.year.astype('int')              # Convierte a entero
sc=MinMaxScaler()                                     # Inicializa el escalador MinMax
final1[["year1"]]=sc.fit_transform(final1[['year']])  # Escala la columna 'year' y la guarda como 'year1'

# Elimina las columnas innecesarias: el año original, rating y avg_rat
final2=final1.drop(columns=['year','rating','avg_rat'])
final2

# Convierte el título de las películas en columnas dummy (one-hot encoding)
final3=pd.get_dummies(final2,columns=['title'])
final3

# Función para obtener películas recomendadas basadas en una película dada 
# Mediante la técnica de correlación la cual se basa en las características (o "contenido") de las películas, como género, año y título, para encontrar películas similares a una película dada.
def recomendacion(movie=list(final1['title'])[0]):
    ind_movie = final1[final1['title'] == movie].index.values.astype(int)[0]      # Encuentra el índice de la película dada
    similar_movie = final3.corrwith(final3.iloc[ind_movie, :], axis=1)            # Calcula la correlación con otras películas
    similar_movie = similar_movie.sort_values(ascending=False)                    # Ordena por correlación descendente
    top_similar_movie = similar_movie.to_frame(name="correlación").iloc[0:11, ]   # Selecciona las 10 películas más similares
    top_similar_movie['title'] = final1["title"]                                  # Añade títulos a las recomendaciones

    return top_similar_movie # Devuelve las películas recomendadas

# Interfaz de usuario para elegir una película y obtener recomendaciones
interact(recomendacion)













#######################################################################
# 3. Sistema de recomendacion basado en el contenido de cada usuario


# Reutilizar el DataFrame 'final' previamente cargado
final

# Eliminar la columna 'userId' para quedarse solo con los datos de las películas
df_usuario = final.loc[:,~final.columns.isin(['userId'])]

# Guardar el DataFrame resultante en la base de datos como 'df_final'
df_usuario.to_sql('df_final', conn, if_exists='replace')

# Consulta SQL para agrupar por título y año y calcular la calificación promedio
movies = pd.read_sql("""select *, avg(rating) as avg_rat
                    from df_final
                    group by year, title
                    order by year desc, avg_rat desc""", conn)

# Eliminar columnas innecesarias
movies = movies.drop(columns=['index','rating','avg_rat'])

# Convertir la columna 'year' a tipo entero
movies['year'] = movies['year'].astype('int')

# Escalar el año para que tenga valores entre 0 y 1, lo que facilita la comparación y mejora el rendimiento del algoritmo
sc=MinMaxScaler() 
sc=MinMaxScaler()
movies[["year_sc"]]=sc.fit_transform(movies[['year']])

# Eliminar columnas que no se utilizarán en el proceso de recomendación
movies1 = movies.drop(columns=['movieId', 'title', 'year'])
movies1

# Extraer los usuarios únicos de la tabla 'ratings_final'
usuarios=pd.read_sql('select distinct (userId) as user_id from ratings_final',conn)


# Función para recomendar películas basadas en el perfil de un usuario específico
def usuario(user_id=list(usuarios['user_id'].value_counts().index)):

    ## Seleccionar solo las calificaciones del usuario seleccionado
    ratings=pd.read_sql('select *from ratings_final where userId=:user',conn, params={'user':user_id})
    ## Convertir los ID de las películas calificadas por el usuario a un array
    l_movies_r=ratings['movieId'].to_numpy()

    ## Agregar columnas 'movieId' y 'title' para poder filtrar y mostrar el nombre de las películas
    movies1[['movieId','title']]=movies[['movieId','title']]
    # Filtrar solo las películas que el usuario ya ha calificado
    movies_r=movies1[movies1['movieId'].isin(l_movies_r)]
    movies_r=movies_r.drop(columns=['movieId','title'])
    movies_r["indice"]=1
    # Calcular el "centroide" del usuario, que es el perfil promedio del usuario basado en sus calificaciones
    centroide=movies_r.groupby("indice").mean()

    # Filtrar las películas que el usuario no ha calificado aún
    movies_nr=movies1[~movies1['movieId'].isin(l_movies_r)]
    movies_nr=movies_nr.drop(columns=['movieId','title'])
    # Usar el modelo NearestNeighbors para encontrar las películas más cercanas al centroide del usuario
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movies_nr)
    dist, idlist = model.kneighbors(centroide)
    # Seleccionar las películas recomendadas basadas en sus índices
    ids=idlist[0]
    recomend_m=movies.loc[ids][['title','movieId']]
    vistos=movies[movies['movieId'].isin(l_movies_r)][['title','movieId']]

    return recomend_m

# Interfaz de usuario para elegir un usuario y obtener recomendaciones de películas basadas en su perfil
interact(usuario)








#######################################################################
# 4. Sistemas de recomendación de filtros colaborativos

# Cargar los ratings de la base de datos
movie_ratings = pd.read_sql('select * from ratings_final', conn)
movie_ratings

# Configurar un objeto 'Reader' para indicar la escala de las calificaciones (de 0 a 5)
reader = Reader(rating_scale=(0, 5))

# Crear un conjunto de datos de Surprise usando las columnas de usuario, ítem y calificación
data = Dataset.load_from_df(movie_ratings[['userId','movieId','rating']], reader)

# Calcular y mostrar el rating promedio de todos los ratings
avg_rating = pd.read_sql('select avg(rating) from ratings_final', conn)

# Definir modelos de vecinos más cercanos a evaluar
models = [KNNBasic(), KNNWithMeans(), KNNWithZScore(), KNNBaseline()]
results = {}


# Bucle para entrenar y evaluar cada modelo usando validación cruzada y almacenar los resultados en un diccionario
for model in models:
    # Validación cruzada de 5 pliegues, evaluando MAE y RMSE en cada pliegue
    CV_scores = cross_validate(model, data, measures=["MAE", "RMSE"], cv=5)
    
    # Calcular la media de los resultados y renombrar las columnas para mejor claridad
    result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
    
    # Extraer el nombre del modelo para usar como clave en el diccionario de resultados
    model_name = str(model).split("algorithms.")[1].split("object ")[0]
    results[model_name] = result


# Convertir los resultados en un DataFrame y ordenar por RMSE
performance_df = pd.DataFrame.from_dict(results).T
sorted_performance = performance_df.sort_values(by='RMSE')

# Definir la grilla de parámetros para la búsqueda de hiperparámetros
param_grid = {
    'sim_options': {
        'name': ['msd', 'cosine'],
        'min_support': [5],
        'user_based': [False, True]
    }
}

# Inicializar y ejecutar la búsqueda de hiperparámetros con validación cruzada de 2 pliegues
grid_search = GridSearchCV(KNNBaseline, param_grid, measures=['rmse'], cv=2)
grid_search.fit(data)

### Mostrar los mejores parámetros y su respectivo RMSE

# Indica los mejores hiperparámetros encontrados por GridSearchCV para el modelo KNNBaseline
print(grid_search.best_params["rmse"]) # = {'sim_options': {'name': 'msd', 'min_support': 5, 'user_based': False}}
# Este resultado indidica que la mejor métrica de similitud para este modelo y conjunto de datos es la "Mean Squared Difference" (Diferencia Cuadrática Media)
# Ademas en 'min_support': 5: Este es el número mínimo de items comunes necesarios entre usuarios para considerarlos "similares"
# Finalmente 'user_based': False: Indica que el modelo basado en vecinos más cercanos (k-NN) que se está utilizando es basado en items y no en usuarios. 

# Este el error promedio del modelo en predicciones de calificaciones, bajo la configuración de hiperparámetros óptima identificada por la búsqueda de cuadrícula
print(grid_search.best_score["rmse"]) # = 0.924400921350208
# Y indica que, en promedio, las predicciones del modelo se desvían aproximadamente 0.92 unidades de las calificaciones reales, en la escala de calificación utilizada

best_model = grid_search.best_estimator['rmse']


# Entrenar el mejor modelo con todos los datos
full_trainset = data.build_full_trainset()
trained_model = best_model.fit(full_trainset)

# Crear un conjunto de prueba con todas las combinaciones usuario-ítem que no están en el conjunto de entrenamiento
testset = full_trainset.build_anti_testset()
predicted_values = trained_model.test(testset)

# Convertir las predicciones en un DataFrame para facilitar el manejo
predicted_df = pd.DataFrame(predicted_values)

# Mostrar algunas estadísticas y predicciones
print(predicted_df.sort_values(by='est', ascending=False))

# Función para recomendar las top N películas a un usuario específico
def get_recommendations(user_id, n_recommendations=10):
    # Filtrar las predicciones para el usuario deseado y obtener las top N
    user_predictions = predicted_df[predicted_df['uid'] == user_id].sort_values(by="est", ascending=False).head(n_recommendations)
    
    # Convertir las recomendaciones en un DataFrame y guardarlo como una tabla en SQL
    top_movies = user_predictions[['iid', 'est']]
    top_movies.to_sql('recommendations', conn, if_exists="replace")
    
    # Unir las recomendaciones con los nombres de las películas y devolver el resultado
    movie_details = pd.read_sql('''select a.*, b.title from recommendations a left join movies b on a.iid=b.movieId''', conn)
    return movie_details

# Obtener y mostrar 15 recomendaciones para el usuario 100
user_recommendations = get_recommendations(user_id=100, n_recommendations=15)
user_recommendations
