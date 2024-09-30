import os, sys 
sys.path.append(os.path.dirname(__file__))
import numpy as np
from deflation import deflation


#Computa autovalores y autovectores de la matriz de covarianza X, si p es -1 computa todos los autovalores/autovectores
def pca(X, p = -1):
    #dimension de CADA COLUMNA O SEA CANTIDAD DE FILAS X COLUMNA
    n = len(X)
    #  CANTIDAD DE COLUMNAS, SI HAY p CANTIDAD DE AUTOVALORES A BUSCAR
    num = len(X[0])
    if p != -1:
        num = p 

    #resto media POR COLUMNAS!!!!
    X_centrada = X - X.mean(0) 
    #esta matriz resulta en hacer el producto interno entre las columnas de X_centrada, es la matriz de covarianza de LAS COLUMNAS  o sea LOS PIXELES
    C = X_centrada.T @ X_centrada / (n-1)   
    
    autovalores, autovectores = deflation(C, num, 20000, 1e-9)
    D = np.diag(autovalores)
    V = autovectores

    return autovalores, autovectores



#Chequear si existe un precomputo de pca con las columnas y el nombre indicado
#Si hay lee del archivo precomputado, sino corre PCA y almacena el resultado
def pca_cache(X, cols, name):
    path = "cache"
    autovalores_path = path + "/autovalores_pca_" + str(cols) + "_" + str(name) + ".csv"
    autovectores_path = path + "/autovectores_pca_" + str(cols) + "_" + str(name) + ".csv"

    if os.path.isfile(autovectores_path) and name != "no_cache":
        V = np.loadtxt(autovectores_path , delimiter=",")
        autovalores = np.loadtxt( autovalores_path, delimiter=",")
    else:
        autovalores, V = pca(X)
        np.savetxt(autovectores_path, V, delimiter=",")
        np.savetxt(autovalores_path, autovalores, delimiter=",")

    return autovalores, V

