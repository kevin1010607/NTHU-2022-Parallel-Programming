#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

const int INF = 1073741823;
void input(char *inFileName);
void output(char *outFileName);
void block_FW(int B);
__global__ void phase1(int B, int r, int *Dist_GPU, int n);
__global__ void phase2(int B, int r, int *Dist_GPU, int n);
__global__ void phase3(int B, int r, int *Dist_GPU, int n, int start);

int n, m, n_origin;
int *Dist, *Dist_GPU[2];

int main(int argc, char* argv[]){
    input(argv[1]);
    int B = 64;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char *inFileName){
    FILE *file = fopen(inFileName, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    n_origin = n;
    n += 64-((n%64+64-1)%64+1);
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
#pragma omp parallel num_threads(2)
{
    int id = omp_get_thread_num(), round = n/64;
    int start = (round/2)*id, size = (round/2)+(round%2)*id;
    cudaSetDevice(id);
    cudaMalloc(&Dist_GPU[id], n*n*sizeof(int));
#pragma omp barrier
    cudaMemcpy(Dist_GPU[id]+(start*64*n), Dist+(start*64*n), size*64*n*sizeof(int), cudaMemcpyHostToDevice);
    for(int r = 0; r < round; r++){
        int copy = (r>=start && r<start+size);
        cudaMemcpyPeer(Dist_GPU[!id]+(r*64*n), !id, Dist_GPU[id]+(r*64*n), id, copy*64*n*sizeof(int));
#pragma omp barrier
        phase1 <<<1, dim3(32, 32)>>> (B, r, Dist_GPU[id], n);
        phase2 <<<dim3(2, round-1), dim3(32, 32)>>> (B, r, Dist_GPU[id], n);
        phase3 <<<dim3(size, round-1), dim3(32, 32)>>> (B, r, Dist_GPU[id], n, start);
    }
    cudaMemcpy(Dist+(start*64*n), Dist_GPU[id]+(start*64*n), size*64*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(Dist_GPU[id]);
}
}

__global__ void phase1(int B, int r, int *Dist_GPU, int n){
    __shared__ int s[64*64];
    int b_i = r<<6, b_j = r<<6, b_k = r<<6;
    int i = threadIdx.y, j = threadIdx.x;

    s[i*64+j] = Dist_GPU[(b_i+i)*n+(b_j+j)];
    s[i*64+(j+32)] = Dist_GPU[(b_i+i)*n+(b_j+(j+32))];
    s[(i+32)*64+j] = Dist_GPU[(b_i+(i+32))*n+(b_j+j)];
    s[(i+32)*64+(j+32)] = Dist_GPU[(b_i+(i+32))*n+(b_j+(j+32))];

    #pragma unroll
    for(int k = 0; k < 64; k++){
        __syncthreads();
        s[i*64+j] = min(s[i*64+j], s[i*64+k]+s[k*64+j]);
        s[i*64+(j+32)] = min(s[i*64+(j+32)], s[i*64+k]+s[k*64+(j+32)]);
        s[(i+32)*64+j] = min(s[(i+32)*64+j], s[(i+32)*64+k]+s[k*64+j]);
        s[(i+32)*64+(j+32)] = min(s[(i+32)*64+(j+32)], s[(i+32)*64+k]+s[k*64+(j+32)]);
    }

    Dist_GPU[(b_i+i)*n+(b_j+j)] = s[i*64+j];
    Dist_GPU[(b_i+i)*n+(b_j+(j+32))] = s[i*64+(j+32)];
    Dist_GPU[(b_i+(i+32))*n+(b_j+j)] =  s[(i+32)*64+j];
    Dist_GPU[(b_i+(i+32))*n+(b_j+(j+32))] = s[(i+32)*64+(j+32)];
}

__global__ void phase2(int B, int r, int *Dist_GPU, int n){
    __shared__ int s[2*64*64];
    // ROW: (blockIdx.x = 1), COL: (blockIdx.y = 0)
    int b_i = (blockIdx.x*r+(!blockIdx.x)*(blockIdx.y+(blockIdx.y>=r)))<<6;
    int b_j = (blockIdx.x*(blockIdx.y+(blockIdx.y>=r))+(!blockIdx.x)*r)<<6;
    int b_k = r<<6;
    int i = threadIdx.y, j = threadIdx.x;

    int val0 = Dist_GPU[(b_i+i)*n+(b_j+j)];
    int val1 = Dist_GPU[(b_i+i)*n+(b_j+(j+32))];
    int val2 = Dist_GPU[(b_i+(i+32))*n+(b_j+j)];
    int val3 = Dist_GPU[(b_i+(i+32))*n+(b_j+(j+32))];

    s[i*64+j] = Dist_GPU[(b_i+i)*n+(b_k+j)];
    s[i*64+(j+32)] = Dist_GPU[(b_i+i)*n+(b_k+(j+32))];
    s[(i+32)*64+j] = Dist_GPU[(b_i+(i+32))*n+(b_k+j)];
    s[(i+32)*64+(j+32)] = Dist_GPU[(b_i+(i+32))*n+(b_k+(j+32))];

    s[4096+i*64+j] = Dist_GPU[(b_k+i)*n+(b_j+j)];
    s[4096+i*64+(j+32)] = Dist_GPU[(b_k+i)*n+(b_j+(j+32))];
    s[4096+(i+32)*64+j] = Dist_GPU[(b_k+(i+32))*n+(b_j+j)];
    s[4096+(i+32)*64+(j+32)] = Dist_GPU[(b_k+(i+32))*n+(b_j+(j+32))];

    __syncthreads();
    #pragma unroll
    for(int k = 0; k < 64; k++){
        val0 = min(val0, s[i*64+k]+s[4096+k*64+j]);
        val1 = min(val1, s[i*64+k]+s[4096+k*64+(j+32)]);
        val2 = min(val2, s[(i+32)*64+k]+s[4096+k*64+j]);
        val3 = min(val3, s[(i+32)*64+k]+s[4096+k*64+(j+32)]);
    }

    Dist_GPU[(b_i+i)*n+(b_j+j)] = val0;
    Dist_GPU[(b_i+i)*n+(b_j+(j+32))] = val1;
    Dist_GPU[(b_i+(i+32))*n+(b_j+j)] = val2;
    Dist_GPU[(b_i+(i+32))*n+(b_j+(j+32))] = val3;
}

__global__ void phase3(int B, int r, int *Dist_GPU, int n, int start){
    __shared__ int s[2*64*64];
    int b_i = (start+blockIdx.x)<<6, b_j = (blockIdx.y+(blockIdx.y>=r))<<6, b_k = r<<6;
    int i = threadIdx.y, j = threadIdx.x;

    int val0 = Dist_GPU[(b_i+i)*n+(b_j+j)];
    int val1 = Dist_GPU[(b_i+i)*n+(b_j+(j+32))];
    int val2 = Dist_GPU[(b_i+(i+32))*n+(b_j+j)];
    int val3 = Dist_GPU[(b_i+(i+32))*n+(b_j+(j+32))];

    s[i*64+j] = Dist_GPU[(b_i+i)*n+(b_k+j)];
    s[i*64+(j+32)] = Dist_GPU[(b_i+i)*n+(b_k+(j+32))];
    s[(i+32)*64+j] = Dist_GPU[(b_i+(i+32))*n+(b_k+j)];
    s[(i+32)*64+(j+32)] = Dist_GPU[(b_i+(i+32))*n+(b_k+(j+32))];

    s[4096+i*64+j] = Dist_GPU[(b_k+i)*n+(b_j+j)];
    s[4096+i*64+(j+32)] = Dist_GPU[(b_k+i)*n+(b_j+(j+32))];
    s[4096+(i+32)*64+j] = Dist_GPU[(b_k+(i+32))*n+(b_j+j)];
    s[4096+(i+32)*64+(j+32)] = Dist_GPU[(b_k+(i+32))*n+(b_j+(j+32))];

    __syncthreads();
    #pragma unroll
    for(int k = 0; k < 64; k++){
        val0 = min(val0, s[i*64+k]+s[4096+k*64+j]);
        val1 = min(val1, s[i*64+k]+s[4096+k*64+(j+32)]);
        val2 = min(val2, s[(i+32)*64+k]+s[4096+k*64+j]);
        val3 = min(val3, s[(i+32)*64+k]+s[4096+k*64+(j+32)]);
    }

    Dist_GPU[(b_i+i)*n+(b_j+j)] = val0;
    Dist_GPU[(b_i+i)*n+(b_j+(j+32))] = val1;
    Dist_GPU[(b_i+(i+32))*n+(b_j+j)] = val2;
    Dist_GPU[(b_i+(i+32))*n+(b_j+(j+32))] = val3;
}