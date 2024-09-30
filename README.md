# FashionMNIST Classifier

## Integrantes
- Josemaría Montaron - [pepemontaron@gmail.com](mailto:pepemontaron@gmail.com)
- Magalí Plotek - [magaliplotek@gmail.com](mailto:magaliplotek@gmail.com)
- Santiago Pivato - [santiagopivato@gmail.com](mailto:santiagopivato@gmail.com)


## Descripción
El objetivo de este proyecto es implementar un algoritmo de ML que clasifique imágenes de prendas de ropa según su tipo, utilizando KNN, 5-folding, PCA y cross-validation. El proyecto se realizó en el marco de la materia Métodos Numéricos de la FCEN-UBA en el primer cuatrimestre del 2024. \
En el [informe](./informe.pdf) se puede encontrar una documentación más extensa de la implementación, junto a varios experimentos y conclusiones.


## Ejecución
El script principal del proyecto explora los hiperparámetros $k$ (knn) y $p$ (PCA) sobre un dataset de entrenamiento, elige la mejor combinación y calcula la precisión promedio en un dataset de prueba. \
Para correr este script, primero se debe compilar el código de c++ con el comando:
```bash 
make 
```
Notar que esto requiere tener instalada la biblioteca Eigen de c++. \
Una vez hecho esto, el comando para correr el script es:
```bash 
python3 run.py 
```
Una vez finalizada la ejecución se imprimirán los $k$ y $p$ óptimos, y la exactitud promedio que estos proveen en el dataset de prueba.

## Probar clasificador
El proyecto cuenta con una [notebook](./classify.ipynb) donde se puede seleccionar una imagen y ver cual es su clase real y cual es la clase que predice el modelo. Para ello se deben ejecutar los dos primeros bloques de código de la notebook en orden.


