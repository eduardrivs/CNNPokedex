# Importamos las librerias necesarias
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Creamos variables globales para parametros futuros
altura, longitud = 100, 100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arreglo = cnn.predict(x) #Este es un arreglo bidimensional [[1,0,0]]
    resultado = arreglo[0]
    respuesta = np.argmax(resultado) #Regresa el indice del arreglo con mayor valor
    if respuesta==0:
        print('Charmander')
    elif respuesta==1:
        print('Gengar')
    elif respuesta==2:
        print('Pikachu')
    return respuesta

predict('pika4.png')