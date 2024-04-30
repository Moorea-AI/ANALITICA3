-- Tabla de películas con el año
DROP TABLE IF EXISTS movies_with_year;
CREATE TABLE movies_with_year AS
    SELECT 
        movieId,
        title,
        genres,
        CAST(SUBSTR(title, -5, 4) AS INTEGER) AS year  -- Extraemos el año del título y lo convertimos a un entero
    FROM movies;

-- Tabla de películas seleccionadas con 5 o más calificaciones
DROP TABLE IF EXISTS movies_sel;
CREATE TABLE movies_sel AS
    SELECT 
        title,
        COUNT(*) AS rating_count
    FROM movies_with_year
    INNER JOIN ratings ON movies_with_year.movieId = ratings.movieId
    GROUP BY title
    HAVING rating_count >= 5
    ORDER BY rating_count DESC;

drop table if exists usuarios_sel;

create table usuarios_sel as 

        SELECT userId, COUNT(*) AS n_id
        FROM ratings
        GROUP BY userId
        HAVING n_id >=10 and n_id <=600
        ORDER BY n_id asc ;



-- Base de datos final de películas
DROP TABLE IF EXISTS movies_final;
CREATE TABLE movies_final AS
    SELECT 
        movies_with_year.movieId,
        movies_with_year.title,
        movies_with_year.genres,
        movies_sel.rating_count,
        movies_with_year.year  -- Incluimos la columna de año en la tabla final de películas
    FROM movies_with_year
    INNER JOIN movies_sel ON movies_with_year.title = movies_sel.title;

-- Base de datos final de calificaciones
DROP TABLE IF EXISTS ratings_final;
CREATE TABLE ratings_final AS
    SELECT 
        ratings.userId,
        ratings.movieId,
        ratings.rating
    FROM ratings
    INNER JOIN usuarios_sel ON ratings.userId = usuarios_sel.userId;

-- Base de datos final
DROP TABLE IF EXISTS df_final;
CREATE TABLE df_final AS
    SELECT 
        ratings_final.userId,
        movies_final.movieId,
        movies_final.title,
        movies_final.genres,
        ratings_final.rating,
        movies_final.year  -- Incluimos la columna de año en la tabla final
    FROM movies_final
    INNER JOIN ratings_final ON movies_final.movieId = ratings_final.movieId;
