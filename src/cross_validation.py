import os, sys 
sys.path.append(os.path.dirname(__file__))
import numpy as np
from knn import knn
from pca import pca_cache, pca


#Ejecuta validacion cruzada para devolver la mejor combinacion entre un p de valores_ y un k de valores_k
#Adicionalmente devuelve cual es la exactitud que se logra con estos dos parametros, y todas las exactitudes para combinaciones de los p y k
#Almacena las exactitudes en un archivo a modo de precómputo
def cross_validation(X_train, y_train, valores_k, valores_p ,name):
    use_cache = name != "no_cache"

    #informacion para guardar archivos 
    exactitudes_path = "cache/resultado_validacion_cruzada_" + str(name) +".csv"
    if os.path.isfile(exactitudes_path) and use_cache:
        exactitudes = np.loadtxt(exactitudes_path, delimiter=",")
    else:
        columnas_X_train = X_train.shape[1]    

        cant_k = len(valores_k)    
        cant_p = len(valores_p)    
        exactitudes = np.zeros((cant_p, cant_k), dtype=np.float64)
        for i in range(0,5):
            print("cross_validation: Procesando fold: " + str(i))                

            X_dev = X_train[i * 1000 : (i+1) * 1000]
            X_newtrain = np.concatenate((X_train[0: i * 1000], X_train[(i+1) * 1000: 5000]))
            y_dev = y_train[i * 1000 : (i+1) * 1000]
            y_newtrain = np.concatenate((y_train[0: i * 1000], y_train[(i+1) * 1000: 5000]))
            
            X_dev_centrada = X_dev - X_dev.mean(axis=1, keepdims=True)
            X_newtrain_centrada = X_newtrain - X_newtrain.mean(axis=1, keepdims=True)

            if use_cache:
                autovalores, V = pca_cache(X_train, columnas_X_train, i)
            else:
                autovalores, V = pca(X_train)

            for indice_p in range(0, cant_p):
                p = valores_p[indice_p]
                X_dev_hat = X_dev_centrada @ V[:, 0:p]
                X_newtrain_hat = X_newtrain_centrada @ V[:, 0:p]

                

                for indice_k in range(0, cant_k):
                    k = valores_k[indice_k]
                    print("cross_validation: Ejecutando knn con k = " + str(k) + " para " + str(p) + " componentes principales")
                    exactitud =  knn(X_dev_hat, X_newtrain_hat, y_dev, y_newtrain, k)
                    exactitudes[indice_p][indice_k] += exactitud

            print("cross_validation: Fold " + str(i) + " procesado")                

        exactitudes /= 5
        if use_cache:
            np.savetxt(exactitudes_path, exactitudes, delimiter=",")

    mejor_indice_p, mejor_indice_k = np.unravel_index(np.argmax(exactitudes), exactitudes.shape)

    mejor_k = valores_k[mejor_indice_k]
    mejor_p = valores_p[mejor_indice_p]
    mejor_exactitud = exactitudes[mejor_indice_p][mejor_indice_k]

    print("cross_validation: Mejor exactitud: " + str(mejor_exactitud) + " obtenida con p = " + str(mejor_p) + " y k = " + str(mejor_k))


    return mejor_p, mejor_k



#Ejecuta PCA y KNN con los parametros p y k
def get_precision(X_train, X_test, y_train, y_test, p, k, use_cache):
    columnas_X_train = X_train.shape[1]
    if(use_cache):
        autovalores, V = pca_cache(X_train, columnas_X_train, "all")
    else:
        autovalores, V = pca(X_train)

    X_test_centrada = X_test - X_test.mean(axis=1, keepdims=True)
    X_train_centrada = X_train - X_train.mean(axis=1, keepdims=True)

    X_test_hat = X_test_centrada @ V[:,0:p]
    X_train_hat = X_train_centrada @ V[:,0:p]


    return knn(X_test_hat, X_train_hat, y_test, y_train, k)


