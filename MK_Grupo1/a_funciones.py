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


#Ejecuta un archivo SQL dado sobre una conexión de base de datos.

def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo) # Abre el archivo SQL
    sql_as_string=sql_file.read() # Lee el contenido del archivo SQL como una cadena
    sql_file.close # Cierra el archivo
    cur.executescript(sql_as_string)  # Ejecuta las instrucciones SQL utilizando el cursor
    