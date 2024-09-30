import os, sys 
sys.path.append(os.path.dirname(__file__))
import numpy as np
import ctypes



#clase para similificar uso de ctypes
class sharedlib():
    dlclose = ctypes.CDLL(None).dlclose 
    dlclose.argtypes = (ctypes.c_void_p,)

    def __init__(self, path, method, *args):
        self.lib = ctypes.cdll.LoadLibrary(f'./{path}')

        # Se explicitan los tipos de los argumentos para el método deseado
        self.method_object = getattr(self.lib, method)
        self.method_object.argtypes = args

    def __call__(self, *args):
        return self.method_object(*args)

    def unload(self):
        while self.dlclose(self.lib._handle)!=-1:
            pass


#declaracion funciones de c a ser importadas
run_deflation = sharedlib(
    'src/eigen_functions.so',
    'run_deflation',
    ctypes.c_char_p,    #input_path
    ctypes.c_char_p,    #output_path_matrix
    ctypes.c_char_p,    #output_path_vector
    ctypes.c_uint,      #num
    ctypes.c_uint,      #n_iter
    ctypes.c_double,    #epsilon
)



#metodo de la potencia con deflacion
def deflation(matriz, num, n_iter, epsilon):
    #crear inpout file
    if(not os.path.isdir("temp")):
        os.mkdir("temp")
    with open("temp/input.csv", "w") as f:
        primera_linea= str(matriz.shape[0]) + "," + str(matriz.shape[1]) + "\n" #dimensiones de la matriz
        f.write(primera_linea)
        np.savetxt(f, matriz, delimiter=",")
    #parametros tipo string
    input_file = ctypes.create_string_buffer(b"temp/input.csv")
    output_file_matriz = ctypes.create_string_buffer(b"temp/output_matriz.csv")
    output_file_vector = ctypes.create_string_buffer(b"temp/output_vector.csv")

    #llamada a funcion de c
    run_deflation(
        input_file,
        output_file_matriz,
        output_file_vector,
        ctypes.c_uint(num),
        ctypes.c_uint(n_iter),
        ctypes.c_double(epsilon),
    )

    #leyendo output
    autovectores = np.loadtxt("temp//output_matriz.csv", delimiter=",")
    autovalores = np.loadtxt("temp/output_vector.csv", delimiter=",")

    #limpiando archivos
    os.remove("temp/input.csv") 
    os.remove("temp/output_matriz.csv")
    os.remove("temp/output_vector.csv")
    os.removedirs("temp")

    return autovalores, autovectores
    
