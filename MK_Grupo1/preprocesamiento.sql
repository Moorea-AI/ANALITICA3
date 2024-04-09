-- Tabla de usuarios con menos de 750 ratings
DROP TABLE IF EXISTS usuarios_sel;
CREATE TABLE usuarios_sel as
                  SELECT UserId, count(*) as cnt_rat_user
                  FROM ratings
                  GROUP BY userId
                  HAVING cnt_rat_user <=750
                  ORDER BY cnt_rat_user ASC;

-- Tabla de peliculas con 5 o más ratings

DROP TABLE IF EXISTS movies_sel;
CREATE TABLE movies_sel as
                  select title , count(*) as rating_count
                         from movies inner join ratings
                         on movies.movieID = ratings.movieID
                         group by title
                         having rating_count >=5
                         order by rating_count desc;

-- Base de datos movies final

DROP TABLE IF EXISTS movies_final;
CREATE TABLE movies_final as
                  select movieID, movies.title, genres, rating_count
                  from movies inner join movies_sel
                  on movies.title = movies_sel.title;

-- Base de datos ratings final

DROP TABLE IF EXISTS ratings_final;
CREATE TABLE ratings_final as
                  select ratings.userID, movieID, rating
                  from ratings inner join usuarios_sel
                  on ratings.userID = usuarios_sel.userID;

-- Base de datos df_final

DROP TABLE IF EXISTS df_final;
CREATE TABLE df_final as
                  select userID, movies_final.movieID, title, genres, rating
                  from movies_final inner join ratings_final
                  on movies_final.movieID = ratings_final.movieID;