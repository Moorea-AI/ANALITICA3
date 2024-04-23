################################################################
#               SEGUNDO TRABAJO PARA LA MATERIA:               #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE SALUD                             #
#                              POR:                            #
#                                                              #
#                    AURA LUZ MORENO - MOOREA                  #
#                                                              #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################


#Ejecuta un archivo SQL dado sobre una conexión de base de datos.

def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo) # Abre el archivo SQL
    sql_as_string=sql_file.read() # Lee el contenido del archivo SQL como una cadena
    sql_file.close # Cierra el archivo
    cur.executescript(sql_as_string)  # Ejecuta las instrucciones SQL utilizando el cursor
    
    
def storeResults(model, a,b,c,d):
    ML_Model.append(model)
    acc_train.append(round(a, 3))
    acc_test.append(round(b, 3))
    rmse_train.append(round(c, 3))
    rmse_test.append(round(d, 3))
    
knn = KNeighborsRegressor()

param_grid = {'n_neighbors':list(range(1, 31)), 'weights': ['uniform', 'distance']}
