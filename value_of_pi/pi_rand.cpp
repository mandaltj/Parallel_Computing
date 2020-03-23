#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>

//RNG - https://channel9.msdn.com/Events/GoingNative/2013/rand-Considered-Harmful

double calc_pi(int num_trials){
	double num_points_in_circle = 0;

	//srand(time(NULL));
	srand(0);
	for (int i=0; i<num_trials; i++){
		//Generating the x and y coordinates randomly
		//Range is between 0 and 1
		double x = double(rand()) / double(RAND_MAX);
		double y = double(rand()) / double(RAND_MAX);

		//Subtraction by 0.5 is to center the cordinates
		//Now the range is -0.5 to 0.5
		double distance = (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5);
		if (distance <= 0.25){
			num_points_in_circle++;
		}
	}

	double pi = (num_points_in_circle/num_trials)*4;
	std::cout<<"Pi value calculated: " << pi <<'\n';
	return pi;
}


int main(int argc, char *argv[]){
	if (argc == 1){
		std::cout << "Error: Please enter number of trials\n";
		exit(1);
	}

	int num_trials = std::atoi(argv[1]);
	std::cout<<"Number of trials: " << num_trials <<"\n";
	const double reference_pi = 3.14159265358979323846;
	//srand(time(NULL));

	auto start_time = std::chrono::high_resolution_clock::now();
	double caclulated_pi = calc_pi(num_trials);
	auto stop_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
	std::cout << "Time elapsed: " << dur_ms.count() << "ms" << std::endl;

	double error = fabs((reference_pi - caclulated_pi)/reference_pi)*100;
	printf("Relative pi error: %4.2e\n", error);

	return 0;
}
