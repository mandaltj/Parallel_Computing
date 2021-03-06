#include <iostream>
#include <pthread.h>

int * arr;
int sum;
pthread_mutex_t sum_lock;

struct thread_id{
    int id;
};

void *calc_sum(void* i){
    thread_id *tid = (thread_id*) i;
    printf("idx: %d\n", tid->id);
    pthread_mutex_lock(&sum_lock);
    sum += arr[tid->id];
    printf("Thread %d updating sum %d\n", tid->id, sum);
    pthread_mutex_unlock(&sum_lock);
    pthread_exit(0);
}

int main(int argc, char *argv[]){
    int num_threads = std::atoi(argv[1]);
    sum = 0;
    arr = (int *) malloc(num_threads*sizeof(int));
    for (int k=0; k<num_threads; k++){
        arr[k] = k+1;
        //std::cout<<arr[k]<<'\n';
    }

    pthread_t p_threads[num_threads];
    pthread_mutex_init(&sum_lock, NULL);
    thread_id tid[num_threads];

    for (int i=0; i<num_threads; i++){
        //printf("i: %d\n", i);
        tid[i].id = i;
        int status = pthread_create(&p_threads[i], NULL, calc_sum, (void *)&tid[i]);
        if (status != 0){
            std::cout<<"Error in pthread creation!!\n";
            exit(1);
        }
    }

    for (int i=0; i<num_threads; i++){
        pthread_join(p_threads[i], NULL);
    }
    std::cout<<"Sum: "<<sum<<'\n';

    return 0;
}
