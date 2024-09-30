import sys, os
import numpy as np
from src.cross_validation import cross_validation, get_precision

#Tipos de ropa
class_names = ["T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


#Datos de entrenamiento en X_train con sus clases asociadas en y_train
X_train = np.loadtxt( os.path.dirname(__file__) + "/data/X_train.csv", delimiter=",")
y_train = np.loadtxt( os.path.dirname(__file__) + "/data/y_train.csv", delimiter=",").astype(int)
#Idem pero para la parte de test
X_test = np.loadtxt( os.path.dirname(__file__) + "/data/X_test.csv", delimiter=",")
y_test = np.loadtxt( os.path.dirname(__file__) + "/data/y_test.csv", delimiter=",").astype(int)



if __name__ == "__main__":
    valores_p = [2,3,4,5,10,25,80,175,450,784]
    valores_k = range(1,21)

    print("--- FashionMNIST Classifier ---")
    print("El proyecto cuenta con datos precomputados para acelerar el script. Puede elegir no usarlos, lo cual implica una demora de varios minutos para volver a computarlos.")

    user_input = None
    while(user_input not in ["y", "n"]):
        user_input = input("¿Desea utilizar datos precomputados? (y/n) - ").lower()


    if(user_input == "y"):
        use_cache = True
        cache_name = "final"
    else:
        use_cache = False
        cache_name = "no_cache"


    p, k = cross_validation(X_train, y_train, valores_k, valores_p, cache_name)
    exactitud = get_precision(X_train, X_test, y_train, y_test, p, k, use_cache)
    print("Exactitud final: " + str(exactitud))




