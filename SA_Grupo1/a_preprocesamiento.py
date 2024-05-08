import numpy as np
import cv2 ### para leer imagenes jpeg  ### pip install opencv-python
from matplotlib import pyplot as plt ## para gráfciar imágnes
import funciones as fn#### funciones personalizadas, carga de imágenes
import joblib ### para descargar array
import os

# Ahora bien, definimos una función para desplegar las imágenes para saber si están cargando bien
def load_and_display(image_path):
    img = cv2.imread(image_path)
    class_name = image_path.split("/")[2]  # Extract class name from path
    resized_img = cv2.resize(img, (120, 120))  # Resize to 100x100
    num_pixels = np.prod(resized_img.shape)  # Calculate number of pixels
    plt.imshow(resized_img)
    plt.title(f"{class_name} - Shape: {resized_img.shape}, Max: {resized_img.max()}, Min: {resized_img.min()}, Pixels: {num_pixels}")
    plt.show()

############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

# Veamos como seria cada una: 
load_and_display('data/train/NonDemented/nonDem15.jpg')
load_and_display('data/train/VeryMildDemented/verymildDem0.jpg')
load_and_display('data/test/MildDemented/26 (19).jpg')
load_and_display('data/train/ModerateDemented/moderateDem10.jpg')  

#Podemos observar que el shape es igual para todas: 208,176,3
#La intnsidad de pixeles en su máximo esta en 243 y 254
# Y los pixeles estan en 109.824
# 208 en el eje Y y 176 en el eje x


################################################################
######## Código para cargar todas las imágenes #############
####### reducir su tamaño y convertir en array ################
################################################################


width = 120 #tamaño para reescalar imágen
num_classes = 2 #clases variable respuesta
trainpath = 'data/train/'
testpath = 'data/test/'

x_train, y_train = fn.img2data(trainpath, width)
x_test, y_test = fn.img2data(testpath, width)




#### convertir salidas a numpy array ####
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train.shape


np.prod(x_train[1].shape)
y_train.shape


x_test.shape
y_test.shape

# Crear la carpeta "salidas" si no existe
if not os.path.exists("salidas"):
    os.makedirs("salidas")

####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "salidas/x_train.pkl")
joblib.dump(y_train, "salidas/y_train.pkl")
joblib.dump(x_test, "salidas/x_test.pkl")
joblib.dump(y_test, "salidas/y_test.pkl")

