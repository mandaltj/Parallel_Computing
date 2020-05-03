#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <cmath>

#define FIXED_FLOAT(x) std::fixed<<std::setprecision(4)<<(x)

using Matrix = std::vector<std::vector<double>>;

struct points{
    double x;
    double y;
};

template <typename T>
void print_vector(const std::vector<T> & A) {
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

double distance_square(const points & A, const points & B){
    return ((A.x-B.x)*(A.x-B.x)) + ((A.y-B.y)*(A.y-B.y));
}


double kernel_scalar(const struct points & A, const struct points & B,
              const double l1, const double l2){
    double distance_square = (((A.x-B.x)*(A.x-B.x))/(2*l1*l1))
                           + (((A.y-B.y)*(A.y-B.y))/(2*l2*l2));
    return exp(-1.0*distance_square);
}

std::vector<double> kernel_vector(const std::vector<points>& A, const std::vector<points>& B,
                             const double l1, const double l2){
    if(A.size()!=B.size()){
        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
    }
    int n = A.size();
    std::vector<double> result(n);

    for(int i=0; i<n; i++){
        double distance_square = (((A[i].x-B[i].x)*(A[i].x-B[i].x))/(2*l1*l1))
                               + (((A[i].y-B[i].y)*(A[i].y-B[i].y))/(2*l2*l2));
        result[i] = exp(-1.0*distance_square);
    }

    return result;
}


//Function to initialize observer data vector f
//m - Dimension of the matrix
std::vector<double> create_f(int m, const std::vector<points> & XY){
    int n = m*m;
    if(n!=XY.size()){
        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
    }

    std::vector<double> f(n);

    //Using "random" library for random number generation
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);

    points idk_point;
    idk_point.x = 0.25;
    idk_point.y = 0.25;
    double l1 = 2/m;
    double l2 = 2/m;

    //#pragma omp parallel for
    for(int i=0; i<n; i++){
        double d = 0.02*distribution(generator);
        f[i] = d + kernel_scalar(XY[i], idk_point, l1, l2) + (XY[i].x*0.2 + XY[i].y*0.1);
    }
    //Debug print
    //print_vector(f);
    return f;
}

//Function to create our K matrix
//n - Dimension of K(nxn) matrix
//Here n equals m*m for this program
Matrix create_K0(int n, const std::vector<points> & XY,
                  const double l1, const double l2){
    Matrix K0(n, std::vector<double>(n, 0));

    //#pragma omp parallel for
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            K0[i][j] = kernel_scalar(XY[i], XY[j], l1, l2);
        }
    }
    return K0;
}

//Function to create our K matrix
//n - Dimension of K(nxn) matrix
//Here n equals m*m for this program
Matrix create_K(const std::vector<int> & itrain,
                  const Matrix & K0){
    int n = itrain.size();
    Matrix K(n, std::vector<double>(n, 0));

    //#pragma omp parallel for
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            K[i][j] = K0[itrain[i]][itrain[j]];
        }
    }
    return K;
}

Matrix create_k_small(const Matrix & K0, const std::vector<int> & itrain, const std::vector<int> & itest){
    int ntrain = itrain.size();
    int ntest = itest.size();
    Matrix k(ntest, std::vector<double>(ntrain, 0));

    //#pragma omp parallel for
    for(int i=0; i<ntest; i++){
        for(int j=0; j<ntrain; j++){
            k[i][j] = K0[itest[i]][itrain[j]];
        }
    }
    return k;
}

void create_vectors(int n, std::vector<int> & itest, std::vector<int> & itrain, double test_percent){
    std::vector<int> test(n);
    for (int i=0; i<n; i++){
        test[i] = i;
    }

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(test), std::end(test), rng);
    //std::cout<<"test:";
    //print_vector(test);

    int ntest = round(test_percent*n);
    int ntrain = n-ntest;
    int itest_count = 0;
    int itrain_count = 0;
    for (int i=0; i<n; i++){
        if(i<ntest){
            itest[itest_count] = test[i];
            itest_count++;
        }
        else{
            itrain[itrain_count] = test[i];
            itrain_count++;
        }
    }
    //std::cout<<"itest:";
    //print_vector(itest);
    //std::cout<<"itrain:";
    //print_vector(itrain);
}

std::vector<double> create_Lparam(double start, double end, double stride){
    std::vector<double> Lparam;
    double i = start;
    //std::cout<<"start: "<<start<<'\n';
    //std::cout<<"end: "<<end<<'\n';
    while(i<end){
        //std::cout<<i<<'\n';
        Lparam.push_back(i);
        i=i+stride;
    }
    double y =1.0;
    double x=1.0;
    if(y<x){
        std::cout<<"Check\n";
    }
    //print_vector(Lparam);
    return Lparam;
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

std::vector<double> mul_matvec(const Matrix & A, const std::vector<double> & x){
    int n = A.size();
    std::vector<double> y(n, 0);
    //#pragma omp parallel for
    for(int i=0; i<n; i++){
        double sum = 0;
        for(int j=0; j<n; j++){
            sum += A[i][j]*x[j];
        }
        y[i] = sum;
    }
    return y;
}

double MSE(const std::vector<double>& fstar_ref, const std::vector<double>& fstar){
    double error = 0;
    if(fstar_ref.size()!=fstar.size()){
        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
    }

    int n = fstar_ref.size();

    for(int i=0; i<n; i++){
        double single_err = fstar_ref[i]-fstar[i];
        error += single_err*single_err;
    }

    return error;
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
        //#pragma omp parallel for
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
        //#pragma omp parallel for
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
        //#pragma omp parallel for schedule(guided)
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

//==============================================================================
//                    Cholesky factorization
//==============================================================================
//Transpose of Matrix
void transpose(Matrix & U, const Matrix & L){
    int n = U.size();
    //#pragma omp parallel for
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            U[i][j] = L[j][i];
        }
    }
}

void Cholesky_factorization(Matrix & L, Matrix & U){
    int dimension = U.size();
    //unsigned int fops = 0;
    // Decomposing a matrix into Lower Triangular
    for (int j = 0; j < dimension; j++) {
        float sum = 0;
        for (int k = 0; k < j; k++) {
            sum += L[j][k] * L[j][k];
            //fops += 2;
        }
        L[j][j] = sqrt(U[j][j] - sum);
        //fops += 2;
        //#pragma omp parallel for if(j<dimension-100)
        for (int i = j + 1; i < dimension; i++) {
            sum = 0;
            for (int k = 0; k < j; k++) {
                sum += L[i][k] * L[j][k];
        //fops += 2;
            }
            L[i][j] = (1.0 / L[j][j] * (U[i][j] - sum));
        //fops += 2;
        }
    }
    //std::cout<<"Factorization fops: "<<fops<<'\n';
    transpose(U, L);
}


//GPR - Gaussiam Process Regression
//m - dimension of the Matrix
std::vector<double> GPR(const std::vector<points>& XY,
                        const std::vector<double> & ftrain,
                        const std::vector<int> & itest,
                        const std::vector<int> & itrain,
                        const double t,
                        const double l1,
                        const double l2){
    int n=XY.size();

    //Initialize K0
    Matrix K0 = create_K0(n, XY, l1, l2);

    int ntest = itest.size();
    int ntrain = itrain.size();

    Matrix K = create_K(itrain, K0);

    //Calculating the tI+K matrix
    Matrix K_dash = ti_k(K, t);
    //std::cout<<"K_dash size: "<<K_dash.size()<<'\n';

    //Creating the framework for L and U matrix to be passed
    //by reference

    int train_dimension = K_dash.size();
    Matrix L(train_dimension, std::vector<double>(train_dimension, 0));
    Matrix U = K_dash;

    //Initialize k
    Matrix k = create_k_small(K0, itrain, itest);
    //std::cout<<"k: ";
    //print_vector(k);


    //LU factorization function
    Cholesky_factorization(L, U);
    //Cholesky_factorization(L, K_dash, n);

    //print_matrix(K_dash);
    //print_matrix(L);
    //print_matrix(U);

    std::vector<double> f_copy = ftrain;
    //std::cout<<"F-copy size: "<<f_copy.size()<<'\n';

    std::vector<double> y = forw_sub(L, f_copy);
    //std::cout<<"L-size: "<<L.size()<<'\n';
    //std::cout<<"y-size: "<<y.size()<<'\n';
    //std::cout<<"U-size: "<<U.size()<<'\n';
    std::vector<double> z = back_sub(U , y);

    //std::cout<<"z: ";
    //print_vector(k);

    //std::cout<<"k-size: "<<k.size()<<'\n';
    //std::cout<<"z-size: "<<z.size()<<'\n';
    std::vector<double> fstar = mul_matvec(k, z);
    //std::cout<<"Test"<<'\n';

    return fstar;
}

int main(int argc, char *argv[]){
    //Check input arguments
    if (argc < 2){
        std::cout << "Error: Please enter 1) Matrix Dimension; \n";
        exit(1);
    }
    //Get the dimension of the matrix
    int m = std::atoi(argv[1]);

    int n = m*m;

    //Initialize m x m grid of points
    //Although it is supposed to be a grid. I am utilizing a vector
    //structure. Makes the code easy to visualize when compared with
    //the provided MATLAB code
    std::vector<points> XY = create_XY(m);
    //Initialize observed data vector f
    std::vector<double> f = create_f(m, XY);

    double test_percent=0.25;
    int ntest = round(test_percent*n);
    int ntrain = n-ntest;
    std::vector<int> itest(ntest);
    std::vector<int> itrain(ntrain);
    create_vectors(n, itest, itrain, test_percent);

    std::vector<double> ftest_ref(ntest, 0);
    std::vector<double> ftrain(ntrain, 0);
    for(int i=0; i<ntest;i++){
        ftest_ref[i] = f[itest[i]];
    }
    for(int i=0; i<ntrain;i++){
        ftrain[i] = f[itrain[i]];
    }

    double Tparam=0.25;
    std::vector<double> Lparam = create_Lparam(0.1, 0.99, 0.1);
    //print_vector(Lparam);
    //std::cout<<Lparam.size()<<'\n';

    auto start_time = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse (2)
    for(int i=0; i<Lparam.size(); i++){
        for(int j=0; j<Lparam.size(); j++){
            std::vector<double> ftest = GPR(XY, ftrain, itest, itrain, Tparam, Lparam[i], Lparam[j]);
            double error = MSE(ftest_ref, ftest);
            //std::cout<<Lparam.size()*i+j<<": Finished (l1, l2) ="<<Lparam[i]<<" "<<Lparam[j]<<" Error="<<error<<'\n';
            //std::cout<<"\n";
        }
    }
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Time elapsed: " << dur_ms.count() << "ms" << std::endl;

}
