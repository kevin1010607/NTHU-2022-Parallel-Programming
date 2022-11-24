#include <stdio.h>
#include <math.h>
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

	ull now = ceil(sqrtl(R-start*start)), res;

	for(ull x = start; x < start+len; x++){
		ull t = R-x*x;
		while(t <= (now-1)*(now-1)) now -= 4;
		while(t > now*now) now++;
		pixels += now;
	}

	pixels %= k;

	MPI_Reduce(&pixels, &res, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Finalize();

	if(rank == 0) printf("%llu\n", (4 * res) % k);
	
	return 0;
}
