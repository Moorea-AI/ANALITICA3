import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Conexión a la base de datos SQLite
conn = sqlite3.connect('data/db_movies')

# Cargar datos
movies = pd.read_sql_query("SELECT * FROM movies", conn)
ratings = pd.read_sql_query("SELECT * FROM ratings", conn)

# Limpieza inicial
# Eliminar posibles duplicados
movies.drop_duplicates(subset='movieId', inplace=True)
ratings.drop_duplicates(subset=['userId', 'movieId'], inplace=True)

# Verificar y manejar valores faltantes si es necesario
# Por ejemplo, podemos decidir eliminar filas con valores faltantes en ciertas columnas
ratings.dropna(subset=['rating'], inplace=True)

print(movies.head())
print(ratings.head())


# Distribución de calificaciones
ratings['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribución de Calificaciones')
plt.xlabel('Calificación')
plt.ylabel('Frecuencia')
plt.show()

# Número de calificaciones por película
ratings_per_movie = ratings.groupby('movieId').size()
ratings_per_movie.hist(bins=30)
plt.title('Distribución del Número de Calificaciones por Película')
plt.xlabel('Número de Calificaciones')
plt.ylabel('Número de Películas')
plt.show()

# Número de calificaciones por usuario
ratings_per_user = ratings.groupby('userId').size()
ratings_per_user.hist(bins=30)
plt.title('Distribución del Número de Calificaciones por Usuario')
plt.xlabel('Número de Calificaciones')
plt.ylabel('Número de Usuarios')
plt.show()


# Filtrar usuarios y películas con un mínimo de interacciones
min_ratings_user = 50
min_ratings_movie = 20

filtered_users = ratings['userId'].value_counts() >= min_ratings_user
filtered_users = filtered_users[filtered_users].index.tolist()

filtered_movies = ratings['movieId'].value_counts() >= min_ratings_movie
filtered_movies = filtered_movies[filtered_movies].index.tolist()

# Aplicar filtro
df_filtered = ratings[(ratings['userId'].isin(filtered_users)) & (ratings['movieId'].isin(filtered_movies))]

print(df_filtered.head())


# Películas más populares basadas en el número de calificaciones
popular_movies = df_filtered['movieId'].value_counts().head(10)
print("Películas más populares:\n", popular_movies)

# Películas más populares basadas en el número de calificaciones
popular_movies = df_filtered['movieId'].value_counts().head(10)
print("Películas más populares:\n", popular_movies)


# Este es un ejemplo básico que recomienda películas basadas en la similitud del género.

# Convertir la lista de géneros en un conjunto de datos binarios
from sklearn.feature_extraction.text import CountVectorizer

# Creando un nuevo dataframe de películas filtradas
filtered_movies_info = movies[movies['movieId'].isin(filtered_movies)]

# Vectorización de los géneros
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = vectorizer.fit_transform(filtered_movies_info['genres'])

# Calculando la similitud del coseno entre películas basadas en géneros
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Función para recomendar películas basadas en similitud de géneros
def recommend_movies_based_on_genre(movie_title, cosine_sim=cosine_sim):
    # Obtener el índice de la película que coincide con el título
    idx = filtered_movies_info.index[filtered_movies_info['title'] == movie_title].tolist()[0]
    
    # Obtener las puntuaciones de similitud de todas las películas con esa película
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordenar las películas basadas en las puntuaciones de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtener las puntuaciones de las 10 películas más similares
    sim_scores = sim_scores[1:11]
    
    # Obtener los índices de las películas
    movie_indices = [i[0] for i in sim_scores]
    
    # Retornar el top 10 de películas más similares
    return filtered_movies_info['title'].iloc[movie_indices]

# Ejemplo de recomendación
print(recommend_movies_based_on_genre("Toy Story (1995)"))
