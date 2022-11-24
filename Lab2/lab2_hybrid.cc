#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#define ull unsigned long long 
#define min(a,b) (((a)<(b))?(a):(b))

int main(int argc, char** argv) {
	ull r = atoll(argv[1]), R = r*r;
	ull k = atoll(argv[2]);
	ull pixels = 0;

	int size, rank;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	ull unit = r/size, remain = r%size;
    ull start = unit*rank+min(remain, rank);
    ull len = unit+(rank < remain);

	ull res = 0;

	int size_omp = 4;
	ull unit_omp = len/size_omp, remain_omp = len%size_omp;

	#pragma omp parallel for num_threads(4) schedule(static, 1) reduction(+:pixels)
	for(int i = 0; i < 4; i++){
		ull start_omp = start+unit_omp*i+min(remain_omp, i);
		ull len_omp = unit_omp+(i < remain_omp);

		ull now_omp = ceil(sqrtl(R-start*start));

		for(ull x = start_omp; x < start_omp+len_omp; x++){
			ull t = R-x*x;
			while(t <= (now_omp-1)*(now_omp-1)) now_omp -= 4;
			while(t > now_omp*now_omp) now_omp++;
			pixels += now_omp;
		}
		pixels %= k;
	}

	MPI_Reduce(&pixels, &res, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Finalize();

	if(rank == 0) printf("%llu\n", (4 * res) % k);
	
	return 0;
}