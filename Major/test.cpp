#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

void print_vector(const std::vector<int> & A) {
    std::cout<<"[ ";
    for(auto A_val: A){
        std::cout<<A_val<<" ";
    }
    std::cout<<"]\n";
}

int main(){
	std::vector<int> test(10);
	for (int i=0; i<10; i++){
		test[i] = i;
	}
    auto rng = std::default_random_engine {};
	std::shuffle(std::begin(test), std::end(test), rng);
	print_vector(test);
	return 0;
}
