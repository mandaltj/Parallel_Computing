#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

//A nice stacoverflow explanation of static and dynamic scheduling
//https://stackoverflow.com/questions/10850155/whats-the-difference-between-static-and-dynamic-schedule-in-openmp

using Matrix = std::vector<std::vector<int>>;

Matrix create_matrix(int dimension){
    Matrix A(dimension, std::vector<int>(dimension, 0));

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 9);

    for(int i=0; i<dimension;i++){
        for(int j=0; j<dimension;j++){
            A[i][j] = distribution(generator);;
        }
    }
    return A;
}

std::vector<int> create_vector(int dimension){
    std::vector<int> A(dimension, 0);

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 9);

    for(int i=0; i<dimension;i++){
        A[i] = distribution(generator);;
    }
    return A;
}


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

void mul_serial(const Matrix & A, const std::vector<int> & x, std::vector<int> & y){
    int dimension = x.size();
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            y[i] += A[i][j]*x[j];
        }
    }
}

//In this static scheduling each thread updates the variable y[i] which is 4 bytes of data. Let's say that cache line
//is 64B then y[0], y[1], y[2] and y[3] will be in the same cache line(provided that the memory allocation for
//std::vector y was 64B aligned in the address space). Now, when Thread 0 in Core 0 updates y[0], it invalidates
//the cache line in Core 1 where Thread 1 is running. Now when Thread 1 tries to access y[1] it gets a miss. This is
//false sharing.

void mul_parallel_static(const Matrix & A, const std::vector<int> & x, std::vector<int> & y, const int chunk_size){
    int dimension = x.size();
    #pragma omp parallel
    {
        printf("%d\n", omp_get_num_threads());

        #pragma omp for schedule(static, chunk_size)
        for(int i=0; i<dimension; i++){
            for(int j=0; j<dimension; j++){
                y[i] += A[i][j]*x[j];
            }
        }

    }
}

//This is a modification from the above function. Here I have used a "sum" variable to store the sum of the dot product
//that is being calculated by a single thread. Take a look at the reports "mat_vec.9517432" and "mat_vec.9517433".
//We can observe that this function outperforms the previous function, because the number of times the invalidation occurs
//has been drastically reduced. Each thread calculates its own sum and then writes y[i] only once. This is a major improvement.
void mul_parallel_static_test(const Matrix & A, const std::vector<int> & x, std::vector<int> & y, const int chunk_size){
    int dimension = x.size();
    #pragma omp parallel
    {
        printf("%d\n", omp_get_num_threads());

        #pragma omp for schedule(static, chunk_size)
        for(int i=0; i<dimension; i++){
            int sum = 0;
            for(int j=0; j<dimension; j++){
                sum += A[i][j]*x[j];
            }
            y[i] = sum;
        }

    }
}

void mul_parallel_dynamic(const Matrix & A, const std::vector<int> & x, std::vector<int> & y, const int chunk_size){
    int dimension = x.size();
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            y[i] += A[i][j]*x[j];
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
    int thread_cnt = std::atoi(argv[2]);
    #pragma omp_set_num_threads(thread_cnt);

    Matrix A = create_matrix(dimension);
    std::vector<int> x = create_vector(dimension);
    std::vector<int> y(dimension, 0);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    // auto start_time = std::chrono::high_resolution_clock::now();
    // mul_serial(A, x, y);
    // auto stop_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Serial: " << dur_ms.count() << "ms" << std::endl;

    //=============================================================================================
    //                            Static Schedule
    //=============================================================================================

    start_time = std::chrono::high_resolution_clock::now();
    mul_parallel_static(A, x, y, 1);
    stop_time = std::chrono::high_resolution_clock::now();
    dur_ms = stop_time - start_time;
    std::cout << "Time elapsed Parallel(static, 1): " << dur_ms.count() << "ms" << std::endl;

    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static(A, x, y, 4);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(static, 4): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static(A, x, y, 8);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(static, 8): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static(A, x, y, 16);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(static, 16): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static(A, x, y, 32);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(static, 32): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static(A, x, y, 64);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(static, 64): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static(A, x, y, 256);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(static, 256): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static(A, x, y, 512);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(static, 512): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static(A, x, y, 1024);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(static, 1024): " << dur_ms.count() << "ms" << std::endl;

    //=============================================================================================
    //                    Static Schedule with private sum variable
    //=============================================================================================
    //
//    start_time = std::chrono::high_resolution_clock::now();
//    mul_parallel_static_test(A, x, y, 1);
//    stop_time = std::chrono::high_resolution_clock::now();
//    dur_ms = stop_time - start_time;
//    std::cout << "Time elapsed Parallel_test(static, 1): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static_test(A, x, y, 4);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel_test(static, 4): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static_test(A, x, y, 8);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel_test(static, 8): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static_test(A, x, y, 16);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel_test(static, 16): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static_test(A, x, y, 32);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel_test(static, 32): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static_test(A, x, y, 64);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel_test(static, 64): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static_test(A, x, y, 256);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel_test(static, 256): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static_test(A, x, y, 512);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel_test(static, 512): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_static_test(A, x, y, 1024);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel_test(static, 1024): " << dur_ms.count() << "ms" << std::endl;

    //=============================================================================================
    //                            Dynamic Schedule
    //=============================================================================================
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_dynamic(A, x, y, 1);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(dynamic, 1): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_dynamic(A, x, y, 4);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(dynamic, 4): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_dynamic(A, x, y, 8);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(dynamic, 8): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_dynamic(A, x, y, 16);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(dynamic, 16): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_dynamic(A, x, y, 32);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(dynamic, 32): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_dynamic(A, x, y, 64);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(dynamic, 64): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_dynamic(A, x, y, 256);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(dynamic, 256): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_dynamic(A, x, y, 512);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(dynamic, 512): " << dur_ms.count() << "ms" << std::endl;
    //
    // start_time = std::chrono::high_resolution_clock::now();
    // mul_parallel_dynamic(A, x, y, 1024);
    // stop_time = std::chrono::high_resolution_clock::now();
    // dur_ms = stop_time - start_time;
    // std::cout << "Time elapsed Parallel(dynamic, 1024): " << dur_ms.count() << "ms" << std::endl;

    return 0;
}
