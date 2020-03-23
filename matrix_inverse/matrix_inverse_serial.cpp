#include <iostream>
#include <vector>
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

void create_triangle_matrix(Matrix & A){
    //Create an Upper Triangular Matrix with random integers
    unsigned int seed = 1;
    int dimension = A.size();
    for(int i=0; i<dimension;i++){
        for(int j=0; j<dimension;j++){
            if(i+j >= 2*i){
                double x = double(rand_r(&seed)%10);
                //Just to avoid Zeroes; Some random number
                if (x==0){
                    A[i][j] = double(rand_r(&seed)%2)+1;
                }
                else{
                    A[i][j] = x;
                }
            }
            seed *= (j+1);
        }
    }

    std::vector<double> temp;
    int sum;
    for(int i=0; i<dimension;i++){
        sum = 0;
        for(int j=0; j<dimension;j++){
            sum += A[j][i];
        }
        temp.push_back(sum);
    }
    if (temp.size()!=dimension){
        throw std::runtime_error(std::string("Error Check\n"));
    }
    for(int i=0; i<dimension;i++){
        A[i][i] += temp[i];
    }
}

Matrix create_identity_matrix(int dimension){
    Matrix ID(dimension, std::vector<double>(dimension, 0));
    for(int i=0; i<dimension;i++){
        for(int j=0; j<dimension;j++){
            if(i+j == 2*i){
                ID[i][j] = 1;
            }
        }
    }
    return ID;
}

double calc_error(const Matrix &A){
    int dimension = A.size();
    Matrix ID = create_identity_matrix(dimension);
    int sum = 0;
    int temp = 0;
    for (int i=0; i<dimension; i++){
        for (int j=0; j<dimension; j++){
            temp = A[i][j]-ID[i][j];
            sum += (temp*temp);
        }
    }
    return sqrt(sum);
}

Matrix multiply_matrix(const Matrix &A, const Matrix &B){
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
    Matrix C(A_row_dimension, std::vector<double>(B_col_dimension, 0));

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

Matrix calc_inverse(const Matrix &A,
                    int start_row,
                    int end_row,
                    int start_col,
                    int end_col) {
    //std::cout<<"Calc inverse Called\n";
    int row_dimension = end_row-start_row+1;
    int col_dimension = end_col-start_col+1;
    //if(row_dimension!=2 || col_dimension!=2){
    //    std::cout<<"row_dimension: "<<row_dimension<<'\n';
    //    std::cout<<"col_dimension: "<<col_dimension<<'\n';
    //    throw std::runtime_error(std::string("Calc Inverse Error: Size != 2\n"));
    //}
    //if(A[end_row][start_col] != 0){
    //    throw std::runtime_error(std::string("Error: Not Upper Triangular Matrix > 2\n"));
    //}
    Matrix B(2, std::vector<double>(2, 0));

    B[0][0] = 1/A[start_row][start_col];
    B[1][1] = 1/A[end_row][end_col];
    B[0][1] = (-1)*(B[0][0]*B[1][1]*A[start_row][end_col]);
    //print_matrix(B);
    return B;
}

void negate_matrix(Matrix &A){
    int row_dimension = A.size();
    int col_dimension = A[0].size();
    //if(row_dimension!=col_dimension){
    //    throw std::runtime_error(std::string("Calc Inverse Error: Size > 2\n"));
    //}
    for(int i=0; i<row_dimension; i++){
        for(int j=0; j<col_dimension; j++){
            A[i][j] = (-1.0) * A[i][j];
        }
    }
}

//Guass Jordan method of Matrix Inverse Calculation
Matrix matrix_inverse_gj(const Matrix &A) {

    //Get the dimension of the matrix
    int dimension = A.size();

    //Create an Identity matrix of equal dimension
    Matrix ID = create_identity_matrix(dimension);

    //Creating a copy of matrix A for inverse calculation
    Matrix A_inv(dimension, std::vector<double>(dimension, 0));
    A_inv = A;

    //==========================================================================
    //                Guass Jordan Method of Matrix Inversion
    //==========================================================================
    //Creating pivots for A[i][i]
    for(int i=0; i<dimension; i++){
        double scale = A_inv[i][i];
        for(int j=0; j<dimension; j++){
            A_inv[i][j]  = A_inv[i][j]/scale;
            ID[i][j] = ID[i][j]/scale;
        }
    }

    //Matrix manipulation of rows
    //This for loop cannot be parallelized
    for(int i=0; i<dimension; i++){
        //The internal for loops can be parallelized because each
        //operation is independent of each other
        for(int row=0; row<dimension; row++){
            if(row == i){
                continue;
            }
            double factor =  A_inv[row][i];
            for(int col=0; col<dimension; col++){
                A_inv[row][col] = A_inv[row][col] - factor*A_inv[i][col];
                ID[row][col] = ID[row][col] - factor*ID[i][col];
            }
        }
    }
    return ID;
}


Matrix matrix_inverse(const Matrix &A,
                        int start_row,
                        int end_row,
                        int start_col,
                        int end_col){
    //std::cout<<"start_row: "<<start_row<<" ";
    //std::cout<<"end_row: "<<end_row<<" ";
    //std::cout<<"start_col: "<<start_col<<" ";
    //std::cout<<"end_col: "<<end_col<<"\n";
    //std::cout<<"========================================\n";

    //double error = 0;

    //int row_dimension = (end_row-start_row+1);
    //int col_dimension = (end_col-start_col+1);

    //Assertion checks
    //if (row_dimension != col_dimension){
    //    std::cout<<"row_dimension: "<<row_dimension<<'\n';
    //    std::cout<<"col_dimension: "<<col_dimension<<'\n';
    //    throw std::runtime_error(std::string("Dimension Error\n"));
    //}

    int dimension = end_row-start_row+1;
    Matrix A_inv(dimension, std::vector<double>(dimension, 0));
    if (dimension<=16){
        Matrix A_copy(dimension, std::vector<double>(dimension, 0));
        for (int i=0; i<dimension; i++){
            for (int j=0; j<dimension; j++){
                A_copy[i][j] = A[start_row+i][start_col+j];
            }
        }
        //std::cout<<"Test 1\n";
        A_inv = matrix_inverse_gj(A_copy);
        //std::cout<<"Test 2\n";
    }
    else{
        //dimension can be even or odd
        //For odd cases we need to do handling slight differently
        int top_dim;
        int btm_dim;

        if (dimension%2==0){
            top_dim = dimension/2;
            btm_dim = dimension/2;
        }
        else{
            top_dim = dimension/2;
            btm_dim = (dimension/2)+1;
        }
        int corner_row_dim = top_dim;
        int corner_col_dim = btm_dim;

        //Creating Empty Matrices for calculating inverse of sub-problems
        Matrix A_inv_top(top_dim, std::vector<double>(top_dim, 0));
        Matrix A_inv_btm(btm_dim, std::vector<double>(btm_dim, 0));
        Matrix A_corner(corner_row_dim, std::vector<double>(corner_col_dim, 0));
        Matrix A_inv_corner(corner_row_dim, std::vector<double>(corner_col_dim, 0));

        A_inv_top = matrix_inverse(A, start_row, start_row+(top_dim-1), start_col, start_col+(top_dim-1));

        A_inv_btm = matrix_inverse(A, start_row+top_dim, end_row, start_col+top_dim, end_col);


        for(int i=0; i<corner_row_dim; i++){
            for(int j=0; j<corner_col_dim; j++){
                A_corner[i][j] = A[start_row+i][start_col+top_dim+j];
            }
        }

        A_inv_corner = multiply_matrix(A_inv_top, A_corner);
        A_inv_corner = multiply_matrix(A_inv_corner, A_inv_btm);
        negate_matrix(A_inv_corner);

        for (int i=0; i<top_dim; i++){
            for (int j=0; j<top_dim; j++){
                A_inv[i][j] = A_inv_top[i][j];
            }
        }

        for (int i=0; i<btm_dim; i++){
            for (int j=0; j<btm_dim; j++){
                A_inv[top_dim+i][top_dim+j] = A_inv_btm[i][j];
            }
        }

        for (int i=0; i<corner_row_dim; i++){
            for (int j=0; j<corner_col_dim; j++){
                A_inv[i][top_dim+j] = A_inv_corner[i][j];
            }
        }

        //std::cout<<"A_inv:\n";
        //print_matrix(A_inv);

    }
    return A_inv;

}



int main(int argc, char *argv[]){
    //Check input arguments
    if (argc < 2){
        std::cout << "Error: Please enter 1) Matrix Dimension; \n";
        exit(1);
    }
    int dimension = std::atoi(argv[1]);
    Matrix A(dimension, std::vector<double>(dimension,0));
    Matrix A_inv(dimension, std::vector<double>(dimension,0));
    Matrix test_matrix(dimension, std::vector<double>(dimension,0));

    create_triangle_matrix(A);
    //print_matrix(A);

    auto start_time = std::chrono::high_resolution_clock::now();
    A_inv = matrix_inverse(A, 0, dimension-1, 0, dimension-1);

    //std::cout<<"Inverse Matrix: \n";
    //print_matrix(A_inv);
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Time elapsed: " << dur_ms.count() << "ms" << std::endl;

    //=======================================================
    //Uncomment this code for Error Check
    //=======================================================
    //test_matrix = multiply_matrix(A, A_inv);
    //std::cout<<"Test Matrix: \n";
    //print_matrix(test_matrix);
    //double error = calc_error(test_matrix);
    //std::cout<<"Error: "<<error<<'\n';

    return 0;

}
