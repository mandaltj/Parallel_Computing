#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){
	int npes, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

	printf("Processor Name: %s; From process %d out of %d, Hello World!\n", processor_name, myrank, npes);
	MPI_Finalize();
}
