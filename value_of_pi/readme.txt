This program calculates the value of pi using the monte carlo method. The file
pi_rand.cpp contains a simple implementaion of the pi calculation. The
pi_rand_parallel.cpp contains a parallelized verison of the code using pthread.

1) Compile command - g++ -o pi_rand_parallel pi_rand_parallel.cpp -lpthread
2) Run Command example:
./pi_rand_parallel 1000000000 1 - Runs with 1 thread
./pi_rand_parallel 1000000000 100 - Runs with 100 thread
3) Observations: This is with my Personal PC
Run with 1 thread:
Number of trials: 1000000000
Number of threads: 1
Time elapsed: 16789.3ms
Pi value calculated: 3.142
Relative pi error: 7.18e-04

Run with 100 threads:
Number of trials: 1000000000
Number of threads: 100
Time elapsed: 2600.25ms
Pi value calculated: 3.142
Relative pi error: 2.70e-03

Speedup = Time taken with 1 thread/ Time taken with 100 threads
		= 16789.3/2600.25
		= 6.456

Efficiency 	= Speedup/Number of threads
			= 6.456/100
			= 6.456%

4) Observations: This is with Texas A&M ADA cluster (non-batch mode)
Run with 1 thread:
Number of trials: 1000000000
Number of threads: 1
Time elapsed: 19945.5ms
Pi value calculated: 3.142
Relative pi error: 7.18e-04

Run with 100 threads:
Number of trials: 1000000000
Number of threads: 100
Time elapsed: 2157.49ms
Pi value calculated: 3.142
Relative pi error: 2.70e-03

5) Observations: This is with Texas A&M ADA cluster (batch mode)
