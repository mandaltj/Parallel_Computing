#include <iostream>
#include <vector>
#include <chrono>
#include <math.h>

using Matrix = std::vector<std::vector<int>>;

void print_matrix(Matrix & A) {
    int dimension = A.size();
    for (int i=0; i<dimension; i++){
        std::cout << "[";
        for (int j=0; j<dimension; j++){
            std::cout << A[i][j];
            if (j!=dimension-1){
                std::cout <<",";
            }
        }
        std::cout << "]\n";
    }
    std::cout<<'\n';
}


std::vector<std::vector<int>> create_matrix(int dimension){
    //Create a Matrix with random integers
    std::vector<std::vector<int>> A(dimension, std::vector<int>(dimension, 0));
    unsigned int seed = 1;
    for(int i=0; i<dimension;i++){
        for(int j=0; j<dimension;j++){
            int x = int(rand_r(&seed)%10);
            //Just to avoid Zeroes; Some random number
            if (x==0){
                A[i][j] = int(rand_r(&seed)%2)+1;
            }
            else{
                A[i][j] = x;
            }
            seed *= (j+1);
        }
    }
    return A;
}

Matrix multiply_matrix_serial(const Matrix &A, const Matrix &B){
    int A_row_dimension = A.size();
    int A_col_dimension = A[0].size();
    int B_row_dimension = B.size();
    int B_col_dimension = B[0].size();
    //if (A_col_dimension!=B_row_dimension){
    //    std::cout<<"A_col_dimension: "<<A_col_dimension<<'\n';
    //    std::cout<<"B_row_dimension: "<<B_row_dimension<<'\n';
    //    throw std::runtime_error(std::string("Multiply Error: Dimension not same\n"));
    //}
    //std::cout<<"A row_dimension:"<<A.size()<<" A col_dimension: "<<A[0].size()<<'\n';
    //std::cout<<"B row_dimension:"<<B.size()<<" B col_dimension: "<<B[0].size()<<'\n';
    Matrix C(A_row_dimension, std::vector<int>(B_col_dimension, 0));

    for(int i=0; i<A_row_dimension; i++){
        for(int j=0; j<B_col_dimension; j++){
            for(int k=0; k<A_col_dimension; k++){
                C[i][j] += A[i][k]*B[k][j];
                //std::cout<<"A["<<i<<"]["<<k<<"]: "<< A[i][k]<<" ";
                //std::cout<<"B["<<k<<"]["<<j<<"]: "<< B[k][j]<<" ";
                //std::cout<<"C["<<i<<"]["<<j<<"]: "<< C[i][j]<<'\n';
            }
        }
    }
    //}
    //std::cout<<'\n';
    return C;
}

Matrix multiply_matrix_parallel(const Matrix &A, const Matrix &B){
    int A_row_dimension = A.size();
    int A_col_dimension = A[0].size();
    int B_row_dimension = B.size();
    int B_col_dimension = B[0].size();
    //if (A_col_dimension!=B_row_dimension){
    //    std::cout<<"A_col_dimension: "<<A_col_dimension<<'\n';
    //    std::cout<<"B_row_dimension: "<<B_row_dimension<<'\n';
    //    throw std::runtime_error(std::string("Multiply Error: Dimension not same\n"));
    //}
    //std::cout<<"A row_dimension:"<<A.size()<<" A col_dimension: "<<A[0].size()<<'\n';
    //std::cout<<"B row_dimension:"<<B.size()<<" B col_dimension: "<<B[0].size()<<'\n';
    Matrix C(A_row_dimension, std::vector<int>(B_col_dimension, 0));

    #pragma omp parallel for collapse(2)
    for(int i=0; i<A_row_dimension; i++){
        for(int j=0; j<B_col_dimension; j++){
            for(int k=0; k<A_col_dimension; k++){
                C[i][j] += A[i][k]*B[k][j];
                //std::cout<<"A["<<i<<"]["<<k<<"]: "<< A[i][k]<<" ";
                //std::cout<<"B["<<k<<"]["<<j<<"]: "<< B[k][j]<<" ";
                //std::cout<<"C["<<i<<"]["<<j<<"]: "<< C[i][j]<<'\n';
            }
        }
    }
    //}
    //std::cout<<'\n';
    return C;
}

bool equal_check(const Matrix & A, const Matrix & B){
    int dimension = A.size();
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            if (A[i][j] != B[i][j]){
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char *argv[]){
    //Check input arguments
    if (argc < 2){
        std::cout << "Error: Please enter 1) Matrix Dimension; \n";
        exit(1);
    }
    int dimension = std::atoi(argv[1]);

    std::vector<std::vector<int>> A = create_matrix(dimension);
    //print_matrix(A);
    std::vector<std::vector<int>> B = create_matrix(dimension);
    //print_matrix(B);

    auto start_time = std::chrono::high_resolution_clock::now();
    Matrix C_serial = multiply_matrix_serial(A, B);
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Time elapsed Serial: " << dur_ms.count() << "ms" << std::endl;
    std::cout<<"C_serial\n";
    print_matrix(C_serial);

    start_time = std::chrono::high_resolution_clock::now();
    Matrix C_parallel = multiply_matrix_parallel(A, B);
    stop_time = std::chrono::high_resolution_clock::now();
    dur_ms = stop_time - start_time;
    std::cout << "Time elapsed Parallel: " << dur_ms.count() << "ms" << std::endl;
    std::cout<<"C_parallel\n";
    print_matrix(C_parallel);

    std::cout<<"Equality Check: "<<equal_check(C_serial, C_parallel)<<'\n';

    return 0;
}
