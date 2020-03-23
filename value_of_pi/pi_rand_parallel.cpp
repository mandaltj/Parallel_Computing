//RNG - https://channel9.msdn.com/Events/GoingNative/2013/rand-Considered-Harmful

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <pthread.h>

#define MAX_THREADS 1000

//The address of glob_val accessed by all threads are same
//int glob_val = 100;

struct thread_data{
    int thread_id;
    int trials_per_thread;
    int seed_value;
    int hits_per_thread;
};

void *calc_pi(void *thread_args){
    thread_data *td_pthread = (thread_data *) thread_args;
    unsigned int seed = td_pthread->seed_value;

	//std::cout<<"Global value addres in Thread: "<<&glob_val<<'\n';
    //rand_r() is thread safe. Other random functions won't work
    for (int i=0; i<td_pthread->trials_per_thread; i++){
        //Generating the x and y coordinates randomly
        //Range is between 0 and 1
        double x = double(rand_r(&seed)) / double(RAND_MAX);
        double y = double(rand_r(&seed)) / double(RAND_MAX);

        //Subtraction by 0.5 is to center the cordinates
        //Now the range is -0.5 to 0.5
        double distance = (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5);
        if (distance <= 0.25){
            td_pthread->hits_per_thread++;
        }
        seed *= (i+1);
    }
    pthread_exit(NULL);
}


int main(int argc, char *argv[]){

    //==========================================================================
    //Check input arguments
    if (argc < 3){
        std::cout << "Error: Please enter 1) Number of trials; 2) Number of threads\n";
        exit(1);
    }
    int num_trials = std::atoi(argv[1]);
    int num_threads = std::atoi(argv[2]);
    if (num_threads > MAX_THREADS){
        std::cout << "Error: Number of threads > MAX_THREADS\n";
        exit(1);
    }
    //==========================================================================
    std::cout<<"Number of trials: " << num_trials <<"\n";
    std::cout<<"Number of threads: " << num_threads <<"\n";
    const double reference_pi = 3.14159265358979323846;

	//std::cout<<"Global value addres: "<<&glob_val<<'\n';

    //==========================================================================
    //                Pthread Code
    //==========================================================================
    pthread_t p_threads[MAX_THREADS];
    pthread_attr_t attr;
    thread_data td[MAX_THREADS];

    int trials_per_thread = num_trials/num_threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i=0; i<num_threads; i++){
        td[i].thread_id = i;
        td[i].trials_per_thread = trials_per_thread;
        td[i].seed_value = i;
        td[i].hits_per_thread = 0;
        int status = pthread_create(&p_threads[i], NULL, calc_pi, (void *)&td[i]);
        if (status != 0){
            std::cout<<"Error in pthread creation!!\n";
            exit(2);
        }
    }

    int total_hits = 0;
    for (int i=0; i<num_threads; i++){
        pthread_join(p_threads[i], NULL);
        total_hits += td[i].hits_per_thread;
    }

    double calculated_pi = ((double)total_hits/(double)num_trials)*4;
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Time elapsed: " << dur_ms.count() << "ms" << std::endl;

    double error = fabs((reference_pi - calculated_pi)/reference_pi)*100;
    printf("Pi value calculated: %4.3f\n", calculated_pi);
    printf("Relative pi error: %4.2e\n", error);

    return 0;
}
