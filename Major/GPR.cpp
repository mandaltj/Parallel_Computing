#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <cmath>

struct points{
    double x;
    double y;
};

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

std::vector<double> kernel_vector(const struct points & A, const struct points & B,
                             const double l1, const double l2){
    if(A.size()!=B.size()){
        throw std::runtime_error(std::string("Error: Dimension mismatch\n"));
    }
    int n = A.size();
    std::vector<double> result(n);

    for(int i=0; i<n; i++){
        double distance_square = (((A.x-B.x)*(A.x-B.x))/(2*l1*l1))
                               + (((A.y-B.y)*(A.y-B.y))/(2*l2*l2));
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

    #pragma omp parallel for
    for(int i=0; i<n; i++){
        double d = 0.02*distribution(generator);
        f[i] = d + kernel_scalar(XY[i], idk_point, l1, l2) + (XY[i].x*0.2 + XY[i].y*0.1);
    }
    //Debug print
    //print_vector(f);
    return f;
}

void create_vectors(int n, std::vector<int> & itest, std::vector<int> & itrain){
	std::vector<int> test(n);
	for (int i=0; i<n; i++){
		test[i] = i;
	}
    auto rng = std::default_random_engine {};
	std::shuffle(std::begin(test), std::end(test), rng);
	int ntest = round(0.1*n);
	int ntrain = n-ntest;
	for (int i=0; i<n; i++){
		if(i<ntest){
			itest[i] = test[i];
		}
		else{
			itrain[i] = test[i];
		}
	}
}

std::vector<double> create_Lparam(double start, double end, douple stride){
	std::vector<double> Lparam;
	for(double i=start; i<end; i=i+stride){
		Lparam.push(i);
	}
	print(Lparam);
	return Lparam;
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
        #pragma omp parallel for
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
        #pragma omp parallel for
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
        #pragma omp parallel for
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
	#pragma omp parallel for if(j<dimension-100)
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
double GPR(int m, const struct points rstar){
    int n=m*m;





    //Initialize K
    Matrix K = create_K(m*m, XY);

    //Calculating the tI+K matrix
    Matrix K_dash = ti_k(K, 0.01);

    //Creating the framework for L and U matrix to be passed
    //by reference

    Matrix L(n, std::vector<double>(n, 0));
    Matrix U = K_dash;

    //Initialize k
    std::vector<double> k = create_k(rstar, XY);
    //std::cout<<"k: ";
    //print_vector(k);

    auto start_time = std::chrono::high_resolution_clock::now();
    //LU factorization function
    Cholesky_factorization(L, U);
    //Cholesky_factorization(L, K_dash, n);
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_ms = stop_time - start_time;
    std::cout << "Cholesky Time elapsed: " << dur_ms.count() << "ms" << std::endl;
    //print_matrix(K_dash);
    //print_matrix(L);
    //print_matrix(U);

    std::vector<double> y = forw_sub(L, f);
    std::vector<double> z = back_sub(U , y);

    //std::cout<<"z: ";
    //print_vector(k);

    double fstar = 0;
    #pragma omp parallel for reduction(+:fstar)
    for(int i=0; i<n; i++){
        fstar += k[i]*z[i];
    }

    return fstar;
}

int main(){
    int n = m*m;

	//Initialize m x m grid of points
	//Although it is supposed to be a grid. I am utilizing a vector
	//structure. Makes the code easy to visualize when compared with
	//the provided MATLAB code
	std::vector<points> XY = create_XY(m);

	//Initialize observed data vector f
	std::vector<double> f = create_f(m, XY);

	int ntest = round(0.1*n);
	int ntrain = n-ntest;
	std::vector<int> itest(ntest);
	std::vector<int> itest(ntrain);
	create_vectors(itest, itrain);

	double Tparam=0.25;

	Lparam = create_Lparam(0.1, 1, 0.1);

	for(int i=0; i<Lparam.size(); i++){
		for(int j=0; j<Lparam.size(); i++){
			ftest = GPR(XY, f, itest, itrain, Tparam, Lparam[i], Lparam[j]);
			double error = f(itest) - ftest;	
		}
	}


}
