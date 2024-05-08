import numpy as np
from os import listdir ### para hacer lista de archivos en una ruta
from tqdm import tqdm  ### para crear contador en un for para ver evolución
from os.path import join ### para unir ruta con archivo 
import cv2 ### para leer imagenes jpg


def img2data(path, width=100):
    
    rawImgs = []   #### una lista con el array que representa cada imágen
    labels = [] ### el label de cada imágen
    
    list_labels = [path+f for f in listdir(path)] ### crea una lista de los archivos en la ruta (Normal /Pneumonia)
    class_names = [folder.split("/")[-1] for folder in list_labels] 

    for imagePath in ( list_labels): ### recorre cada carpeta de la ruta ingresada
        
        files_list=listdir(imagePath) ### crea una lista con todos los archivos
        for item in tqdm(files_list):  
            file = join(imagePath, item) 
            if file[-1] == 'g':  
                img = cv2.imread(file)  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                img = cv2.resize(img, (width, width))  
                rawImgs.append(img) 
                
                # Get label index based on class name
                label_index = class_names.index(imagePath.split("/")[-1]) 
                labels.append(label_index)

    return rawImgs, labels








def img2data2(path, width=100):
    
    rawImgs = []   #### una lista con el array que representa cada imágen
    labels = [] ### el label de cada imágen
    
    list_labels = [join(path, f) for f in listdir(path)] ### crea una lista de los archivos en la ruta (Normal /Pneumonia)

    for imagePath in list_labels: ### recorre cada carpeta de la ruta ingresada
        
        files_list = listdir(imagePath) ### crea una lista con todos los archivos
        for item in tqdm(files_list):  
            file = join(imagePath, item) 
            if file[-1] == 'g':  
                img = cv2.imread(file)  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                img = cv2.resize(img, (width, width))  
                rawImgs.append(img) 
                
                # Clasificar en Alzheimer (1) o No Alzheimer (0)
                if "Demented" in imagePath:
                    labels.append(1)
                else:
                    labels.append(0)

    return rawImgs, labels