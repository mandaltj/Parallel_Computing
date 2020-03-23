#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <math.h>
#include <omp.h>

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
//                    LU factorization
//==============================================================================
void LU_factorization(Matrix & L, Matrix & U){
    int dimension = L.size();

    //This for loop walks through every column of matrix
    for(int i=0; i<dimension; i++){
        //The internal for loops can be parallelized because each
        //operation is independent of each other
        #pragma omp parallel for
        for(int row=i+1; row<dimension; row++){
            double factor = U[row][i]/U[i][i];
            for(int col=i; col<dimension; col++){
                U[row][col] = U[row][col] - factor*U[i][col];
                L[row][i] = factor;
            }
        }
    }
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
    Matrix L = create_identity_matrix(dimension);

    //Create copy of matrix A which will be modified to upper matrix
    Matrix U = A;

    auto start_time = std::chrono::high_resolution_clock::now();
    LU_factorization(L, U);
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Time elapsed Parallel: " << dur_ms.count() << "ms" << std::endl;

    //Matrix A_check = multiply_matrix(L, U);
    //std::cout<<"Error: "<<calc_error(A, A_check)<<"\n";

    return 0;
}
