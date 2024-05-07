import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')


#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') 

x_train /=255 
x_test /=255


##### cargar modelo  ######

modelo=tf.keras.models.load_model('path_to_my_model.h5')



prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

# threshold_no_alz=0.508

# pred_test=(modelo.predict(x_test)>=0.508).astype('int')
# print(metrics.classification_report(y_test, pred_test))
# cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
# disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Pneu', 'Normal'])
# disp.plot()



# Obtener las probabilidades de cada clase para cada muestra
y_pred_prob = modelo.predict(x_test)

# Obtener las clases predichas
y_pred_classes = np.argmax(y_pred_prob, axis=1)

# Crear la matriz de confusión
conf_matrix = metrics.confusion_matrix(y_test, y_pred_classes)

# Mostrar la matriz de confusión en forma de heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3'], yticklabels=['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3'])
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Real')
plt.title('Matriz de Confusión')
plt.show()








### desempeño en entrenamiento #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train=(prob>=threshold_alz).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Pneu', 'Normal'])
disp.plot()


########### ##############################################################
####desempeño en evaluación para grupo 1 (No tienen Alzheimer) #######
########### ##############################################################

prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold


threshold_no_neu=0.5015

pred_test=(modelo.predict(x_test)>=threshold_no_neu).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Alz', 'Normal'])
disp.plot()



### desempeño en entrenamiento #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train=(prob>=threshold_no_neu).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Pneu', 'Normal'])
disp.plot()



####### clasificación final ################

prob=modelo.predict(x_test)

clas=['Neu' if prob >0.508 else 'No Neu' if prob <0.5015 else "No ident" for prob in prob]

clases, count =np.unique(clas, return_counts=True)

count*100/np.sum(count)