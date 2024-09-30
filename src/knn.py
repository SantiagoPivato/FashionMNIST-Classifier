import numpy as np
from sklearn.preprocessing import normalize
from scipy import stats



#implementacion optimizada con cuentas matriciales
def knn(X_dev, X_newtrain, y_dev, y_newtrain, k):
    A = X_dev - X_dev.mean(axis = 1, keepdims = True)
    B = X_newtrain - X_newtrain.mean(axis = 1, keepdims = True)

    A = normalize(A)
    B = normalize(B)

    distancias = 1 - A@B.T

    indices_ordenados = np.argsort(distancias, axis = 1)

    clases_new_train = y_newtrain[indices_ordenados[:,0:k]]

    moda = stats.mode(clases_new_train, axis = 1).mode

    aciertos = np.logical_and( moda, y_dev)
    return np.sum(aciertos) / len(y_dev)




def knn_get_class(x, X_train, y_train, k):
    a = x - x.mean()
    B = X_train - X_train.mean(axis = 1, keepdims = True)

    a /= np.linalg.norm(a)
    B = normalize(B)

    distancias = 1 - a@B.T

    indices_ordenados = np.argsort(distancias)

    clases_new_train = y_train[indices_ordenados[0:k]]
    print(clases_new_train)

    moda = stats.mode(clases_new_train).mode

    return moda