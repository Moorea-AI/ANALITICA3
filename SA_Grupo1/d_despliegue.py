# ################################################################
# #               SEGUNDO TRABAJO PARA LA MATERIA:               #
# #                 APLICACIONES DE LA ANALITICA                 #
# #                  MÓDULO DE SALUD                             #
# #                              POR:                            #
# #                                                              #
# #                    AURA LUZ MORENO - MOOREA                  #
# #                                                              #
# #                    UNIVERSIDAD DE ANTIOQUIA                  #
# ################################################################

# import numpy as np
# import pandas as pd
# import sqlite3 as sql
# import a_funciones as fn ## para procesamiento
# import openpyxl
# from mlxtend.preprocessing import TransactionEncoder

# ####Paquete para sistema basado en contenido ####
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import neighbors

# conn=sql.connect('data/db_movies')
# cur=conn.cursor()

# fn.ejecutar_sql('preprocesamiento.sql', cur)
# # TENGO QUE HACER UN PREPROCESAMIENTO 2 CON LOS DATOS DE DB_MOVIES 2
# # POR ESTE LADO NO ES

# movies=pd.read_sql('select * from movies_with_year', conn )
# ratings=pd.read_sql('select * from ratings', conn)

# genres=movies['genres'].str.split('|')
# te = TransactionEncoder()
# genres = te.fit_transform(genres)
# genres = pd.DataFrame(genres, columns = te.columns_)
# movies.genres.unique()
# movies_dum1 = pd.concat([movies, genres], axis=1)    

# #SE DEBEN HACER AQUI LOS PROCESAMIENTOS DE JOBLIB

# def preprocesar():

#     #### conectar_base_de_Datos#################
#     conn=sql.connect('db_movies_2')
#     cur=conn.cursor()
    

#     fn.ejecutar_sql('preprocesamiento.sql', cur)

#     movies=pd.read_sql('select * from movies_with_year', conn )
#     ratings=pd.read_sql('select * from ratings_final', conn)
    
#     genres=movies['genres'].str.split('|')
#     te = TransactionEncoder()
#     genres = te.fit_transform(genres)
#     genres = pd.DataFrame(genres, columns = te.columns_)
#     movies.genres.unique()
#     movies_dum1 = pd.concat([movies, genres], axis=1)

#     movies_dum1['year'] = movies_dum1['title'].str.extract(r'\((\d{4})\)')
#     movies_dum1['year']=movies_dum1.year.astype('int')
#     sc=MinMaxScaler()
#     movies_dum1[["year"]]=sc.fit_transform(movies_dum1[['year']])
#     movies_dum1=movies_dum1.drop(columns=['genres','title', 'movieId'])


#     col_dum=genres.columns
#     movies_dum2=pd.get_dummies(movies_dum1,columns=col_dum)
#     return movies_dum2,movies, conn, cur

# def recomendar(user_id):
    
#     movies_dum2,movies, conn, cur = preprocesar()

#     ratings=pd.read_sql('select * from ratings_final where userId=:user',conn, params={'user':user_id})
#     l_movie_r=ratings['movieId'].to_numpy() 
#     movies_dum2[['movieId','title']]=movies[['movieId','title']]
#     movies_r=movies_dum2[movies_dum2['movieId'].isin(l_movie_r)]
#     movies_r=movies_r.drop(columns=['movieId','title'])
#     movies_r["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
#     centroide=movies_r.groupby("indice").mean()
    
    
#     movie_nr=movies_dum2[~movies_dum2['movieId'].isin(l_movie_r)]
#     movie_nr=movie_nr.drop(columns=['movieId','title'])
#     model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
#     model.fit(movie_nr)
#     dist, idlist = model.kneighbors(centroide)
    
#     ids=idlist[0]
#     recomend_b=movies.loc[ids][['title','movieId']]

#     return recomend_b

# def main(list_user):
    
#     recomendaciones_todos=pd.DataFrame()
#     for user_id in list_user:
            
#         recomendaciones=recomendar(user_id)
#         recomendaciones["user_id"]=user_id
#         recomendaciones.reset_index(inplace=True,drop=True)
        
#         recomendaciones_todos=pd.concat([recomendaciones_todos, recomendaciones])

#     recomendaciones_todos.to_excel('\\salidas\\recomendaciones\\recomendaciones.xlsx')
#     recomendaciones_todos.to_csv('\\salidas\\recomendaciones\\recomendaciones.csv')
    
# if __name__=="__main__":
#     list_user=[480,250,510,2]
#     main(list_user)
    

# import sys
# sys.executable