#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#define US_PER_SEC 1000000
#define BS 32
#define LBS 5 // 1<<LBS = BS

const int INF = 1073741823;
void input(char *inFileName);
void output(char *outFileName);
void block_FW(int B);
__global__ void phase1(int B, int r, int *Dist_GPU, int n);
__global__ void phase2(int B, int r, int *Dist_GPU, int n);
__global__ void phase3(int B, int r, int *Dist_GPU, int n);

int n, m, n_origin;
int *Dist, *Dist_GPU;

int main(int argc, char* argv[]){
    struct timeval start, end;
    double time;
    gettimeofday(&start, NULL);

    input(argv[1]);
    int B = BS;
    block_FW(B);
    output(argv[2]);

    gettimeofday(&end, NULL);
    time = (double)(US_PER_SEC*(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec))/US_PER_SEC;
    printf("Time: %.2lf\n", time);
    return 0;
}

void input(char *inFileName){
    FILE *file = fopen(inFileName, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    n_origin = n;
    n += BS-((n%BS+BS-1)%BS+1);
    cudaMallocHost(&Dist, n*n*sizeof(int));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            Dist[i*n+j] = (i==j&&i<n_origin)?0:INF;
        }
    }

    int pair[3];
    for(int i = 0; i < m; i++){
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName){
    FILE *file = fopen(outFileName, "w");
    for(int i = 0; i < n_origin; i++){
        fwrite(&Dist[i*n], sizeof(int), n_origin, file);
    }
    fclose(file);
    cudaFreeHost(Dist);
}

void block_FW(int B){
    cudaMalloc(&Dist_GPU, n*n*sizeof(int));
    cudaMemcpy(Dist_GPU, Dist, n*n*sizeof(int), cudaMemcpyHostToDevice);
    int round = n/BS;
    for(int r = 0; r < round; r++){
        phase1 <<<1, dim3(BS, BS)>>> (BS, r, Dist_GPU, n);
        phase2 <<<dim3(2, round-1), dim3(BS, BS)>>> (BS, r, Dist_GPU, n);
        phase3 <<<dim3(round-1, round-1), dim3(BS, BS)>>> (BS, r, Dist_GPU, n);
    }
    cudaMemcpy(Dist, Dist_GPU, n*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(Dist_GPU);
}

__global__ void phase1(int B, int r, int *Dist_GPU, int n){
    int b_i = r<<LBS, b_j = r<<LBS, b_k = r<<LBS;
    int i = threadIdx.x, j = threadIdx.y;

    for(int k = 0; k < BS; k++){
        __syncthreads();
        Dist_GPU[(b_i+i)*n+(b_j+j)] = min(Dist_GPU[(b_i+i)*n+(b_j+j)], Dist_GPU[(b_i+i)*n+(b_k+k)]+Dist_GPU[(b_k+k)*n+(b_j+j)]);
    }
}

__global__ void phase2(int B, int r, int *Dist_GPU, int n){
    // ROW: (blockIdx.x = 1), COL: (blockIdx.y = 0)
    int b_i = (blockIdx.x*r+(!blockIdx.x)*(blockIdx.y+(blockIdx.y>=r)))<<LBS;
    int b_j = (blockIdx.x*(blockIdx.y+(blockIdx.y>=r))+(!blockIdx.x)*r)<<LBS;
    int b_k = r<<LBS;
    int i = threadIdx.x, j = threadIdx.y;

    __syncthreads();
    for(int k = 0; k < BS; k++){
        Dist_GPU[(b_i+i)*n+(b_j+j)] = min(Dist_GPU[(b_i+i)*n+(b_j+j)], Dist_GPU[(b_i+i)*n+(b_k+k)]+Dist_GPU[(b_k+k)*n+(b_j+j)]);
    }
}

__global__ void phase3(int B, int r, int *Dist_GPU, int n){
    int b_i = (blockIdx.x+(blockIdx.x>=r))<<LBS, b_j = (blockIdx.y+(blockIdx.y>=r))<<LBS, b_k = r<<LBS;
    int i = threadIdx.x, j = threadIdx.y;

    __syncthreads();
    for(int k = 0; k < BS; k++){
        Dist_GPU[(b_i+i)*n+(b_j+j)] = min(Dist_GPU[(b_i+i)*n+(b_j+j)], Dist_GPU[(b_i+i)*n+(b_k+k)]+Dist_GPU[(b_k+k)*n+(b_j+j)]);
    }
}