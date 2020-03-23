#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <math.h>

using Matrix = std::vector<std::vector<double>>;

void print_matrix(const Matrix & A) {
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

Matrix create_matrix(int dimension){
    Matrix A(dimension, std::vector<double>(dimension, 0));

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 9);

    for(int i=0; i<dimension;i++){
        for(int j=0; j<dimension;j++){
            A[i][j] = distribution(generator);;
        }
    }
    return A;
}

Matrix create_identity_matrix(int dimension){
    Matrix ID(dimension, std::vector<double>(dimension, 0));
    #pragma omp parallel for
    for(int i=0; i<dimension;i++){
        for(int j=0; j<dimension;j++){
            if(i+j == 2*i){
                ID[i][j] = 1;
            }
        }
    }
    return ID;
}

double calc_error(const Matrix &A, const Matrix &B){
    int dimension = A.size();
    int sum = 0;
    int temp = 0;
    for (int i=0; i<dimension; i++){
        for (int j=0; j<dimension; j++){
            temp = A[i][j]-B[i][j];
            sum += (temp*temp);
        }
    }
    return sqrt(sum);
}

//==============================================================================
//                    Matrix Multiply
//==============================================================================
Matrix multiply_matrix(const Matrix &A, const Matrix &B){
    int A_row_dimension = A.size();
    int A_col_dimension = A[0].size();
    int B_row_dimension = B.size();
    int B_col_dimension = B[0].size();
    Matrix C(A_row_dimension, std::vector<double>(B_col_dimension, 0));

    #pragma omp parallel for
    for(int i=0; i<A_row_dimension; i++){
        for(int j=0; j<B_col_dimension; j++){
            for(int k=0; k<A_col_dimension; k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    return C;
}

//==============================================================================
//                    Matrix Transpose
//==============================================================================
void transpose(Matrix & U, const Matrix & L){
    int n = U.size();
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            U[i][j] = L[j][i];
        }
    }
}


//==============================================================================
//                    Cholesky factorization
//==============================================================================
void Cholesky_factorization_parallel(Matrix & L, Matrix & U){
    int dimension = U.size();
    for (int i = 0; i < dimension; i++) {
        #pragma omp parallel for
        for (int j = 0; j <= i; j++) {
            int sum = 0;
            for (int k = 0; k < j; k++){
                sum += (L[i][k] * L[j][k]);
            }
            L[i][j] = (U[i][j] - sum)/L[j][j];
        }
    }
    transpose(U, L);
}

int main(int argc, char* argv[]){
    //Check input arguments
    if (argc < 2){
        std::cout << "Error: Please enter 1) Dimension; \n";
        exit(1);
    }
    int dimension = std::atoi(argv[1]);

    //Create a square matrix filled with random elements
    Matrix A = create_matrix(dimension);

    //Create Identity matrix which will be modified to lower matrix
    Matrix L(dimension, std::vector<double>(dimension, 0));

    //Create copy of matrix A which will be modified to upper matrix
    Matrix U = A;

    auto start_time = std::chrono::high_resolution_clock::now();
    Cholesky_factorization_parallel(L, U);
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Time elapsed Parallel: " << dur_ms.count() << "ms" << std::endl;

    //Matrix A_check = multiply_matrix(L, U);
    //std::cout<<"Error: "<<calc_error(A, A_check)<<"\n";

    return 0;
}
