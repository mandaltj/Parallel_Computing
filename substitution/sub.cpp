#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

#define FIXED_FLOAT(x) std::fixed<<std::setprecision(4)<<(x)

using Matrix = std::vector<std::vector<double>>;

void print_vector(const std::vector<double> & A) {
    std::cout<<"[ ";
    for(auto A_val: A){
        std::cout<<A_val<<" ";
    }
    std::cout<<"]\n\n";
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

void create_upper_triangle_matrix(Matrix & A){
    //Create an Upper Triangular Matrix with random integers
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(1.0, 9.0);

    int dimension = A.size();
    for(int i=0; i<dimension;i++){
        for(int j=0; j<dimension;j++){
            if(i+j >= 2*i){
                A[i][j] = distribution(generator);
            }
        }
    }

    //std::vector<double> temp;
    //int sum;
    //for(int i=0; i<dimension;i++){
    //    sum = 0;
    //    for(int j=0; j<dimension;j++){
    //        sum += A[j][i];
    //    }
    //    temp.push_back(sum);
    //}
    //if (temp.size()!=dimension){
    //    throw std::runtime_error(std::string("Error Check\n"));
    //}
    //for(int i=0; i<dimension;i++){
    //    A[i][i] += temp[i];
    //}
}

std::vector<double> create_vector(int m){
    std::vector<double> f(m);

    //Using "random" library for random number generation
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(1.0, 9.0);

    for(int i=0; i<m; i++){
        f[i] = distribution(generator);

    }
    //Debug print
    //print_vector(f);
    return f;
}


std::vector<double> back_sub_0 (const Matrix & A, std::vector<double> & y){
    if(A.size()!=y.size()){
        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
    }

    int n = y.size();
    std::vector<double> x(n);

    for(int i = n-1; i>=0; i--){
        double sum = 0;
        for(int j = i+1; j<n; j++){
            sum += A[i][j]*x[j];
        }
        x[i] = (y[i]-sum)/A[i][i];
    }
    return x;
}

std::vector<double> back_sub_1 (const Matrix & A, std::vector<double> & y){
    if(A.size()!=y.size()){
        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
    }

    int n = y.size();
    std::vector<double> x(n);

    for(int i = n-1; i>=0; i--){
        x[i] = y[i]/A[i][i];

        //This loop can be parallelized; Reduction operator needs to be used
        for(int j = i-1; j>=0; j--){
            y[j] -= A[j][i]*x[i];
        }
    }
    return x;
}

int main(int argc, char *argv[]){
    //Check input arguments
    if (argc < 2){
        std::cout << "Error: Please enter 1) Matrix Dimension; \n";
        exit(1);
    }
    //Get the dimension of the matrix
    int m = std::atoi(argv[1]);

    Matrix A(m, std::vector<double>(m,0));
    create_upper_triangle_matrix(A);
    print_matrix(A);

    std::vector<double> y = create_vector(m);
    print_vector(y);

    //==========================================================================
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> x_0 = back_sub_0(A, y);

    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Time elapsed: " << dur_ms.count() << "ms" << std::endl;
    //==========================================================================
    print_vector(x_0);

    //==========================================================================
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> x_1 = back_sub_1(A, y);

    stop_time = std::chrono::high_resolution_clock::now();
    dur_ms = stop_time - start_time;
    std::cout << "Time elapsed: " << dur_ms.count() << "ms" << std::endl;
    //==========================================================================
    print_vector(x_1);

    return 0;
}
