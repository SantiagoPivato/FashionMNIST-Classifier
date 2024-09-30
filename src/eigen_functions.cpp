#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <tuple>

#include <dlfcn.h>



std::pair<double, Eigen::VectorXd> power_iteration(Eigen::MatrixXd Matrix, unsigned int num_iteraciones, double epsilon) {
    int m = Matrix.cols();
    Eigen::VectorXd autovector = Eigen::VectorXd::Random(m); // autovector con el que iniciamos 
    Eigen::VectorXd autovector_pre = Eigen::VectorXd::Random(m); //autovector anterior

    for(int i = 0; i < num_iteraciones; i++) {
        autovector_pre = autovector;
        autovector = Matrix * autovector;
        autovector = autovector / autovector.norm();

        if((autovector-autovector_pre).cwiseAbs().maxCoeff()<epsilon) {
            //std::cout<<"Método de la potencia: Salió por epsilon en la iteracion "<<i<<std::endl;
            break;
        }
    }
    
    double norma = autovector.norm();

    double autovalor = (autovector.transpose().dot(Matrix * autovector)) / (norma*norma);    

    std::pair<double, Eigen::VectorXd> res = std::make_pair(autovalor, autovector);
    
    return res;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> deflation(Eigen::MatrixXd Matrix,unsigned int num,unsigned int num_iteraciones, double epsilon) {
    int n = Matrix.rows();
    int m = Matrix.cols();
    Eigen::VectorXd autovalores = Eigen::VectorXd::Zero(num);
    Eigen::MatrixXd autovectores = Eigen::MatrixXd::Zero(m, num);

    for(int i = 0; i < num; i++) {
        if(i%100 == 0)
            std::cout<<"deflation: Autovalores restantes: "<<num - i<<std::endl;
        std::pair<double, Eigen::VectorXd> par = power_iteration(Matrix, num_iteraciones, epsilon);
        autovalores[i] = par.first;
        autovectores.col(i) = par.second;

        Matrix = Matrix - par.first * par.second * par.second.transpose();
    }
    
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> res = std::make_pair(autovalores, autovectores);
    return res;
}


//Funciones auxiliares para input/output de matrices al interfacear con python
Eigen::MatrixXd leer_matriz(char* path) {
    std::ifstream input;
    input.open(path); //abro el archivo en input
    std::string row;
    std::string row_i;
    int k = 0;
    int size[2];

    int row_n = 0;
    int column_n;

    std::string valor;
    std::getline(input, row_i);
    std::stringstream lineStream(row_i);
    while(std::getline(lineStream,valor,',')) {
        size[k]=stoi(valor);
        k++;
    }
    Eigen::MatrixXd Matrix = Eigen::MatrixXd::Zero(size[0], size[1]); 


    while(std::getline(input, row)) { //en row tengo la fila
        std::stringstream lineStream(row); //la transformo en un strig stream
        column_n = 0;
        while(std::getline(lineStream,valor,',')) { 
            if (column_n >= 0) { 
                Matrix(row_n, column_n) = std::stod(valor); //covierto string a int, genera problema con floats?
            }
            column_n++;
        }
        
        row_n++; //hecho eso incremento las filas
    }

    input.close();
    return Matrix;
}

void escribir_matriz(char* path, Eigen::MatrixXd Matrix) {
    std::ofstream output;
    output.open(path); 
    std::string row;
    int n = Matrix.rows();
    int m = Matrix.cols();
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m - 1; j++){
            output << Matrix.row(i)[j] << ",";
        }
        output << Matrix.row(i)[m - 1] << "\n";
    }
    output.close();
}

void escribir_vector(char* path, Eigen::VectorXd Vector) {
    std::ofstream output;
    output.open(path); 
    std::string row;
    int n = Vector.rows();
    for(int i = 0; i < n; i++){
        output << Vector.row(i) << "\n";  //formato vector columna, tambien podria ser con comas
    }
    output.close();
}



/*

FUNCIONES A EXPROTAR POR USANDO CTYPE

*/
extern "C" {
    void run_deflation(char* input, char* output_m, char* output_v, unsigned int num, unsigned int num_iteraciones, double epsilon) {
        Eigen::MatrixXd Matrix = leer_matriz(input);
        std::pair<Eigen::VectorXd, Eigen::MatrixXd> resultados = deflation(Matrix, num, num_iteraciones, epsilon);
        escribir_matriz(output_m, resultados.second);
        escribir_vector(output_v, resultados.first);
    }
}
