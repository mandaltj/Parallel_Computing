#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define ARR_SIZE 1000000
std::vector<int> arr(ARR_SIZE, 1);

int main(){
    int sum = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    //Without the "reduction" the race condition on sum leads to error
    #pragma omp parallel for reduction(+:sum)
    //#pragma omp parallel for
    for (int i= 0; i<ARR_SIZE; i++){
        sum = sum + arr[i];
    }

    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Time elapsed: " << dur_ms.count() << "ms" << std::endl;

    std::cout<<sum<<'\n';
    return 0;
}
