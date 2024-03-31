
---procesamientos

---crear tabla con usuarios con más de 50 películas calificadas y menos de 1000

DROP TABLE IF EXISTS usuarios_sel;

CREATE TABLE usuarios_sel AS
SELECT "userId" AS user_id, COUNT(*) AS cnt_rat
FROM ratings
GROUP BY "userId"
HAVING cnt_rat > 50 AND cnt_rat <= 1000
ORDER BY cnt_rat DESC;


---crear tabla con películas que han sido calificadas entre 20 y 150 usuarios
DROP TABLE IF EXISTS movies_sel;

CREATE TABLE movies_sel AS
SELECT movieId, COUNT(*) AS cnt_rat
FROM ratings
GROUP BY movieId
HAVING cnt_rat >= 20 AND cnt_rat <= 150
ORDER BY cnt_rat DESC;


---crear tablas filtradas de películas, usuarios y calificaciones ----

DROP TABLE IF EXISTS ratings_final;

CREATE TABLE ratings_final AS
SELECT a."userId" AS user_id,
       a."rating" AS movie_rating,
       a."movieId" AS movieId
FROM ratings a
INNER JOIN movies_sel b ON a.movieId = b.movieId
INNER JOIN usuarios_sel c ON a."userId" = c.user_id; -- Corregido, se cambió "movieId" por "userId"


DROP TABLE IF EXISTS users_final;

CREATE TABLE users_final AS
SELECT a."movieId" AS movieId,
       a."userId" AS userId,
       a."rating" AS rating
FROM ratings a
INNER JOIN usuarios_sel c ON a."userId" = c.user_id; -- Corregido, se cambió "movieId" por "userId"


DROP TABLE IF EXISTS movies_final;

CREATE TABLE movies_final AS
SELECT a."movieId" AS movieId,
       a.title AS title,
       a.genres AS genres
FROM movies a
INNER JOIN movies_sel c ON a."movieId" = c.movieId;


---crear tabla completa ----

DROP TABLE IF EXISTS full_ratings ;

CREATE TABLE full_ratings AS
SELECT
    a.*,
    b.userId,
    c.title,
    c.genres
FROM ratings_final a
INNER JOIN users_final b ON a.userId = b.userId -- Corregido, se cambió "movieId" por "userId"
INNER JOIN movies_final c ON a.movieId = c.movieId;
