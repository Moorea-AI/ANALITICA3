import numpy as np
import pandas as pd
import cv2 ### para leer imagenes jpeg
### pip install opencv-python
import funciones as fn#### funciones personalizadas, carga de imágenes
import tensorflow as tf
import openpyxl

import sys
sys.executable
sys.path

if __name__=="__main__":

    #### cargar datos ####
    path = '/data/despliegue/'
    x, _, files= fn.img2data(path) #cargar datos de despliegue

    x=np.array(x) ##imagenes a predecir

    x=x.astype('float')######convertir para escalar
    x/=255######escalar datos


    files2= [name.rsplit('.', 1)[0] for name in files] ### eliminar extension a nombre de archivo

    modelo=tf.keras.models.load_model('/salidas/best_alzheimers_model.h5') ### cargar modelo
    prob=modelo.predict(x)


    clas=['Neu' if prob >0.508 else 'No Neu' if prob <0.5015 else "No ident" for prob in prob]

    res_dict={
        "paciente": files2,
        "clas": clas   
    }
    resultados=pd.DataFrame(res_dict)

    resultados.to_excel('/salidas/clasificados.xlsx', index=False)
