#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define ARR_SIZE 100000000
std::vector<int> arr(ARR_SIZE, 1);

int do_sum(int start, int end){
	//printf("start:%d; end:%d; thread_id:%d\n", start, end, omp_get_thread_num());
	int mid, x, y;
    int res=0;
    if((end - start)<=1000){
        for(int i=start; i<=end; i++){
            res += arr[start];
        }
    }
    else{
        mid = (start+end)/2;
		#pragma omp task shared(x)
	    x = do_sum(start, mid);
	    #pragma omp task shared(y)
	    y = do_sum(mid+1, end);
	    #pragma omp taskwait
        res = x+y;
    }
    return res;
}

int calc_sum(int start, int end){
	int sum = 0;
	#pragma omp parallel
	{
		#pragma omp single
		sum = do_sum(start, end);
	}


	for(int i =0; i < ARR_SIZE; i++){
		sum += arr[i];
	}
	return sum;
}

int main(){
    int sum = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    sum = calc_sum(0, ARR_SIZE-1);

	auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Time elapsed: " << dur_ms.count() << "ms" << std::endl;

    std::cout<<sum<<'\n';
    return 0;
}
