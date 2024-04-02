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

from surprise import Reader, Dataset
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import cross_validate, GridSearchCV

#### conectar_base_de_Datos

conn=sql.connect('data//db_movies')
cur=conn.cursor()

#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()


# 1. sistemas basados en popularidad 
##### recomendaciones basado en popularidad ######

#¿Cuáles son las 10 películas con la calificación promedio más alta?
consulta_sql = """
    SELECT title, AVG(rating) as calificacion
    FROM full_ratings
    GROUP BY title
    ORDER BY calificacion DESC 
    LIMIT 10
"""
pd.read_sql(consulta_sql, conn)

#### ¿Cuáles son las 10 películas más populares (con más calificaciones) junto con el número total de calificaciones que han recibido?
consulta_sql = """
    SELECT title, COUNT(rating) as total_calificacion
    FROM full_ratings
    GROUP BY title
    ORDER BY total_calificacion DESC 
    LIMIT 10
"""
pd.read_sql(consulta_sql, conn)

####¿Cuáles son los 5 géneros más vistos basado en la cantidad total de películas en ese género?
consulta_sql = """
    SELECT SUBSTR(genres, 1, INSTR(genres, '|') - 1) AS genero, COUNT(genres) AS total_peliculas, AVG(rating) as calificacion
    FROM full_ratings
    GROUP BY genero
    ORDER BY calificacion DESC
    LIMIT 5
"""
pd.read_sql(consulta_sql, conn)

consulta_sql = """
    SELECT * FROM movies_final
"""
pd.read_sql(consulta_sql, conn)



# 2.1 Sistema de recomendación basado en contenido un solo producto - Manual 

movies=pd.read_sql('select * from movies_final', conn )

movies.info()


# escalar para que año esté en el mismo rango

genres=movies['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)
movies.genres.unique()
movies_dum1 = pd.concat([movies, genres], axis=1)
# movies_dum1.info()

# eliminar filas que no se van a utilizar

movies_dum1['year'] = movies_dum1['title'].str.extract(r'\((\d{4})\)')
movies_dum1.info()
movies_dum1['year']=movies_dum1.year.astype('int')
sc=MinMaxScaler()
movies_dum1[["year"]]=sc.fit_transform(movies_dum1[['year']])
movies_dum1.info()
movies_dum1=movies_dum1.drop(columns=['genres','title', 'movieId'])
# movies_dum1.info()


col_dum=genres.columns
movies_dum2=pd.get_dummies(movies_dum1,columns=col_dum)
movies_dum2.shape

joblib.dump(movies_dum2,"data\\moviess_dum2.joblib") ### para utilizar en segundos modelos

# movies_dum2.info()

# movies recomendadas ejemplo para un pelicula

pelicula='Toy Story (1995)'
ind_pelicula=movies[movies['title']==pelicula].index.values.astype(int)[0]
similar_movies=movies_dum2.corrwith(movies_dum2.iloc[ind_pelicula,:],axis=1)
similar_movies=similar_movies.sort_values(ascending=False)
top_similar_movies=similar_movies.to_frame(name="correlación").iloc[0:11,]### el 11 es número de movies recomendados
top_similar_movies['title']=movies["title"] ### agregaro los nombres (como tiene mismo indice no se debe cruzar)
    


# peliculas recomendados ejemplo para visualización todos las peliculas

def recomendacion(pelicula = list(movies['title'])):
     
    ind_pelicula=movies[movies['title']==pelicula].index.values.astype(int)[0]
    similar_movies=movies_dum2.corrwith(movies_dum2.iloc[ind_pelicula,:],axis=1)
    similar_movies=similar_movies.sort_values(ascending=False)
    top_similar_movies=similar_movies.to_frame(name="correlación").iloc[0:11,]### el 11 es número de movies recomendados
    top_similar_movies['title']=movies["title"] ### agregaro los nombres (como tiene mismo indice no se debe cruzar)
      
    return top_similar_movies


print(interact(recomendacion))



# 2.1 Sistema de recomendación basado en contenido KNN un solo producto visto 

# entrenar modelo 

# el coseno de un angulo entre dos vectores es 1 cuando son perpendiculares y 0 cuando son paralelos(indicando que son muy similar324e-06	3.336112e-01	3.336665e-01	3.336665e-es)
model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(movies_dum2)
dist, idlist = model.kneighbors(movies_dum2)

distancias=pd.DataFrame(dist) ## devuelve un ranking de la distancias más cercanas para cada fila(movie)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde


def MovieRecommender(movie_name = list(movies['title'].value_counts().index)):
    movie_list_name = []
    movie_id = movies[movies['title'] == movie_name].index
    movie_id = movie_id[0]
    for newid in idlist[movie_id]:
        movie_list_name.append(movies.loc[newid].title)
    return movie_list_name


print(interact(MovieRecommender))


# 3 Sistema de recomendación basado en contenido KNN 
# Con base en todo lo visto por el usuario 


usuarios=pd.read_sql('select distinct (userId) as user_id from ratings_final',conn)


def recomendar(user_id=list(usuarios['user_id'].value_counts().index)):
    
    ###seleccionar solo los ratings del usuario seleccionado
    ratings=pd.read_sql('select * from ratings_final where userId=:user',conn, params={'user':user_id})
    
    ###convertir ratings del usuario a array
    l_movie_r=ratings['movieId'].to_numpy()
    
    ###agregar la columna de isbn y titulo del libro a dummie para filtrar y mostrar nombre
    movies_dum2[['movieId','title']]=movies[['movieId','title']]
    
    ### filtrar libros calificados por el usuario
    movies_r=movies_dum2[movies_dum2['movieId'].isin(l_movie_r)]
    
    ## eliminar columna nombre e ID
    movies_r=movies_r.drop(columns=['movieId','title'])
    movies_r["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    ##centroide o perfil del usuario
    centroide=movies_r.groupby("indice").mean()
    
    
    ### filtrar libros no leídos
    movie_nr=movies_dum2[~movies_dum2['movieId'].isin(l_movie_r)]
    ## eliminbar nombre e isbn
    movie_nr=movie_nr.drop(columns=['movieId','title'])
    
    ### entrenar modelo 
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(movie_nr)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0] ### queda en un array anidado, para sacarlo
    recomend_b=movies.loc[ids][['title','movieId']]
#    vistos=movies[movies['movieId'].isin(l_movie_r)][['title','movieId']]
    
    return recomend_b

print(interact(recomendar))


# 4 Sistema de recomendación filtro colaborativo 


pd.read_sql('select * from ratings_final where rating>0', conn)

### datos originales en pandas
## knn solo sirve para calificaciones explicitas
ratings=pd.read_sql('select * from ratings_final where rating>0', conn)

####los datos deben ser leidos en un formato espacial para surprise
reader = Reader(rating_scale=(0, 5)) ### la escala de la calificación
###las columnas deben estar en orden estándar: user item rating
data   = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

#Existen varios modelos
models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline()]
results = {}

# función para probar varios modelos 
model=models[1]
for model in models:

    CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1)

    result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
             rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
    results[str(model).split("algorithms.")[1].split("object ")[0]] = result

performance_df = pd.DataFrame.from_dict(results).T
performance_df.sort_values(by='RMSE')

param_grid = {
    'sim_options': {
        'name': ['msd', 'cosine'],
        'min_support': [5],
        'user_based': [False, True]
    },
    'bsl_options': {
        'method': ['als', 'sgd'],  # Opciones para KNNBaseline
    }
}

### se afina si es basado en usuario o basado en ítem

gridsearchKNNBaseline = GridSearchCV(KNNBaseline, param_grid, measures=['rmse'], \
                                      cv=2, n_jobs=2)

gridsearchKNNBaseline.fit(data)

gridsearchKNNBaseline.best_params["rmse"]

gridsearchKNNBaseline.best_score["rmse"]

gs_model=gridsearchKNNBaseline.best_estimator['rmse'] ### mejor estimador de gridsearch
gs_model

# Entrenar con todos los datos y Realizar predicciones con el modelo afinado

trainset = data.build_full_trainset() ### esta función convierte todos los datos en entrnamiento, las funciones anteriores dividen  en entrenamiento y evaluación
model=gs_model.fit(trainset) ## se reentrena sobre todos los datos posibles (sin dividir)

predset = trainset.build_anti_testset() ### crea una tabla con todos los usuarios y los libros que no han leido
# en la columna de rating pone el promedio de todos los rating, en caso de que no pueda calcularlo para un item-usuario
len(predset)

predictions = gs_model.test(predset) ### función muy pesada, hace las predicciones de rating para todos las peliculas que no hay leido un usuario
### la funcion test recibe un test set constriuido con build_test method, o el que genera crosvalidate

predictions_df = pd.DataFrame(predictions) ### esta tabla se puede llevar a una base donde estarán todas las predicciones
predictions_df.shape

predictions_df.head()

predictions_df['r_ui'].unique() ### promedio de ratings
predictions_df.sort_values(by='est',ascending=False)

##### funcion para recomendar los 10 peliculas con mejores predicciones y llevar base de datos para consultar resto de información
def recomendaciones(user_id,n_recomend=10):

    predictions_userID = predictions_df[predictions_df['uid'] == user_id].\
                    sort_values(by="est", ascending = False).head(n_recomend)

    recomendados = predictions_userID[['iid','est']]
    recomendados.to_sql('reco',conn,if_exists="replace")

    recomendados=pd.read_sql('''select a.*, b.title
                             from reco a left join movies_final b
                             on a.iid=b.movieId ''', conn)

    return(recomendados)

recomendaciones(user_id=429,n_recomend=10)