#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <omp.h>

//https://stackoverflow.com/questions/18553210/how-to-implement-matlabs-mldivide-a-k-a-the-backslash-operator

#define FIXED_FLOAT(x) std::fixed<<std::setprecision(4)<<(x)

using Matrix = std::vector<std::vector<double>>;

struct points{
    double x;
    double y;
};

double distance_square(const points & A, const points & B){
    return ((A.x-B.x)*(A.x-B.x)) + ((A.y-B.y)*(A.y-B.y));
}

void print_vector(const std::vector<double> & A) {
    std::cout<<"[ ";
    for(auto A_val: A){
        std::cout<<A_val<<" ";
    }
    std::cout<<"]\n";
}

void print_matrix(const Matrix & A) {
    int dimension = A.size();
    for (int i=0; i<dimension; i++){
        std::cout << "[";
        for (int j=0; j<dimension; j++){
            std::cout << FIXED_FLOAT(A[i][j]);
            if (j!=dimension-1){
                std::cout <<",";
            }
        }
        std::cout << "]\n";
    }
    std::cout<<'\n';
}

//==============================================================================
//                    Matrix Multiplication Function
//==============================================================================
Matrix multiply_matrix(const Matrix &A, const Matrix &B){
    int A_row_dimension = A.size();
    int A_col_dimension = A[0].size();
    int B_row_dimension = B.size();
    int B_col_dimension = B[0].size();
    Matrix C(A_row_dimension, std::vector<double>(B_col_dimension, 0));

    for(int i=0; i<A_row_dimension; i++){
        for(int j=0; j<B_col_dimension; j++){
            for(int k=0; k<A_col_dimension; k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    return C;
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

//=============================================================================
//Function to initialize m x m grid of points
//m - Dimension of the matrix
//Returns a vector containing m*m points, where each point
//has a X and Y coordinate
//=============================================================================
std::vector<points> create_XY(int m){
    //If the dimension is 3, then we will have 9 total points
    int n = m*m;
    std::vector<points> XY(n);
    double h = 1.0/(m+1);
    int idx = 0;
    for(int i=1; i<=m; i++){
        for(int j=1; j<=m; j++){
            XY[idx].x = i*h;
            XY[idx].y = j*h;
            idx++;
        }
    }

    //Debug print
    //std::cout<<"XY: [ ";
    //for(auto XY_point: XY){
    //    std::cout<<"("<<XY_point.x<<","<<XY_point.y<<") ";
    //}
    //std::cout<<"]\n";

    return XY;
}

//Function to initialize observer data vector f
//m - Dimension of the matrix
std::vector<double> create_f(int m, const std::vector<points> & XY){
    int n = m*m;
    std::vector<double> f(n);

    //Using "random" library for random number generation
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);

    #pragma omp parallel for
    for(int i=0; i<n; i++){
        double d = distribution(generator);
        f[i] = 1.0 + d +
        ((XY[i].x-0.5)*(XY[i].x-0.5) + (XY[i].y-0.5)*(XY[i].y-0.5));
    }
    //Debug print
    //print_vector(f);
    return f;
}

//Function to create our K matrix
//n - Dimension of K(nxn) matrix
//Here n equals m*m for this program
Matrix create_K(int n, const std::vector<points> & XY ){
    Matrix K(n, std::vector<double>(n, 0));

    #pragma omp parallel for
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            double d_square = distance_square(XY[i], XY[j]);
            //std::cout<<d_square<<'\n';
            K[i][j] = exp(-1.0*d_square);
        }
    }
    return K;
}

std::vector<double> create_k(const struct points P, const std::vector<points> & XY){
    int n = XY.size();
    std::vector<double> k(n, 0);

    #pragma omp parallel for
    for(int i=0; i<n; i++){
        double d_square = distance_square(P, XY[i]);
        k[i] = exp(-1.0*d_square);
    }
    return k;
}

//==============================================================================
//                            Back substitution
//==============================================================================
//This code for backward substitution is not good for Parallelezing
//This is a naive algorithm and I don't think this can be parallelized
//std::vector<double> back_sub (const Matrix & A, std::vector<double> & y){
//    if(A.size()!=y.size()){
//        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
//    }
//
//    int n = y.size();
//    std::vector<double> x(n);
//
//    for(int i = n-1; i>=0; i--){
//        double sum = 0;
//        for(int j = i+1; j<n; j++){
//            sum += A[i][j]*x[j];
//        }
//        x[i] = (y[i]-sum)/A[i][i];
//    }
//    return x;
//}

//This code suits better for Parallelizing
//This is a different way of back substitution which can be parallelized
std::vector<double> back_sub (const Matrix & A, std::vector<double> & y){
    if(A.size()!=y.size()){
        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
    }

    int n = y.size();
    std::vector<double> x(n);

    for(int i = n-1; i>=0; i--){
        x[i] = y[i]/A[i][i];
        //This loop can be parallelized. No need of reduction operator
        #pragma omp parallel for
        for(int j = i-1; j>=0; j--){
            y[j] -= A[j][i]*x[i];
        }
    }
    return x;
}

//==============================================================================
//                            Forward substitution
//==============================================================================
//This code for forward substitution is not good for Parallelezing
//std::vector<double> forw_sub (const Matrix & A, std::vector<double> & y){
//    if(A.size()!=y.size()){
//        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
//    }
//
//    int n = y.size();
//    std::vector<double> x(n);
//
//    for(int i = 0; i<n; i++){
//        double sum = 0;
//        for(int j = 0; j<i; j++){
//            sum += A[i][j]*x[j];
//        }
//        x[i] = (y[i]-sum)/A[i][i];
//    }
//    return x;
//}

//This loop is better for parallelizing
std::vector<double> forw_sub (const Matrix & A, std::vector<double> & y){
    if(A.size()!=y.size()){
        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
    }

    int n = y.size();
    std::vector<double> x(n);

    for(int i = 0; i<n; i++){
        x[i] = y[i]/A[i][i];
        //This loop can be parallelized; Reduction operator needs to be used
        #pragma omp parallel for
        for(int j = i+1; j<n; j++){
            y[j] -= A[j][i]*x[i];
        }
    }
    return x;
}
//==============================================================================

//==============================================================================
//                    LU factorization
//==============================================================================

void LU_factorization(Matrix & L, Matrix & U, const Matrix & K){
    //std::cout<<"Matrix K: \n";
    //print_matrix(K);

    //It is assumed that K will be a square matrix
    int dimension = K.size();

    //This for loop walks through every column of matrix
    for(int i=0; i<dimension; i++){
        //The internal for loops can be parallelized because each
        //operation is independent of each other
        #pragma omp parallel for
        for(int row=i+1; row<dimension; row++){
            //std::cout<<"row: "<<row<<'\n';
            double factor = U[row][i]/U[i][i];
            //std::cout<<"factor: "<<factor<<'\n';
            for(int col=i; col<dimension; col++){
                U[row][col] = U[row][col] - factor*U[i][col];
            }
            L[row][i] = factor;
            //std::cout<<"Matrix U: \n";
            //print_matrix(U);
        }
    }

    //This is a check if the LU factorization occure correctly or not
    //K_check should be equal to K
    //Matrix K_check = multiply_matrix(L, U);

    //std::cout<<"Matrix K_check: \n";
    //print_matrix(K_check);

    //std::cout<<"Matrix L: \n";
    //print_matrix(L);

    //std::cout<<"Matrix U: \n";
    //print_matrix(U);
}

//This step creates the matrix K' as mentioned in the HW
Matrix ti_k(const Matrix & K, double t){
    int dimension = K.size();
    Matrix K_dash = K;
    for(int x=0; x<dimension; x++){
        K_dash[x][x] = t+K[x][x];
    }
    return K_dash;
}

//GPR - Gaussiam Process Regression
//m - dimension of the Matrix
double GPR(int m, const struct points rstar){
    int n=m*m;

    //Initialize m x m grid of points
    //Although it is supposed to be a grid. I am utilizing a vector
    //structure. Makes the code easy to visualize when compared with
    //the provided MATLAB code
    std::vector<points> XY = create_XY(m);

    //Initialize observed data vector f
    std::vector<double> f = create_f(m, XY);

    //Initialize K
    Matrix K = create_K(m*m, XY);

    //Calculating the tI+K matrix
    Matrix K_dash = ti_k(K, 0.01);

    //Creating the framework for L and U matrix to be passed
    //by reference
    Matrix L = create_identity_matrix(m*m);
    Matrix U = K_dash;

    //Initialize k
    std::vector<double> k = create_k(rstar, XY);
    //std::cout<<"k: ";
    //print_vector(k);


    auto start_time = std::chrono::high_resolution_clock::now();
    //LU factorization function
    LU_factorization(L, U, K_dash);
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "LU elapsed: " << dur_ms.count() << "ms" << std::endl;

    std::vector<double> y = forw_sub(L, f);
    std::vector<double> z = back_sub(U , y);

    //std::cout<<"z: ";
    //print_vector(k);

    double fstar = 0;
    #pragma omp parallel for reduction(+:fstar)
    for(int i=0; i<n; i++){
        fstar += k[i]*z[i];
    }

    return fstar;
}

int main(int argc, char *argv[]){
    //Check input arguments
    if (argc < 4){
        std::cout << "Error: Please enter 1) Matrix Dimension; \n";
        exit(1);
    }
    //Get the dimension of the matrix
    int m = std::atoi(argv[1]);
    points rstar;
    rstar.x = std::atof(argv[2]);
    rstar.y = std::atof(argv[3]);

    double fstar = GPR(m, rstar);

    std::cout<<"fstar: "<<fstar<<'\n';

    return 0;
}
