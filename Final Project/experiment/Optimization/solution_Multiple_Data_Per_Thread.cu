#include <fstream>
#include <cuda.h>
#include <sys/time.h>
#define US_PER_SEC 1000000
#define N 10000
#define NT 1024
// CPU
const char *input_filename, *output_filename;
int n, dp_n, data[N+2], res;
// GPU
__constant__ int data_GPU[N+2];
int *dp;

// CPU
void input();
void output();
void solve();
// GPU
__device__ int get_data(int, int, int, int, int, int*);
__global__ void max_reduce(int, int, int, int, int*);

int main(int argc, char* argv[]){
    struct timeval start, end;
    double time;
    gettimeofday(&start, NULL);

    input_filename = argv[1];
    output_filename = argv[2];

    input();
    solve();
    output();

    gettimeofday(&end, NULL);
    time = (double)(US_PER_SEC*(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec))/US_PER_SEC;
    printf("Time: %.6lf\n", time);
    return 0;
}

void input(){
    // read input file
    std::ifstream infile(input_filename, std::ios::binary);
    infile.read(reinterpret_cast<char*>(&n), sizeof(int));
    infile.read(reinterpret_cast<char*>(&data[1]), n*sizeof(int));
    infile.close();
}

void output(){
    // write output file
    std::ofstream outfile(output_filename, std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(&res), sizeof(int));
    outfile.close();
}

void solve(){
    // solution
    data[0] = data[n+1] = 1;
    dp_n = n+2;
    int NUM = (dp_n+(NT*2)-1)/(NT*2);
    cudaMemcpyToSymbol(data_GPU, data, dp_n*sizeof(int));
    cudaMalloc(&dp, (dp_n+1)*dp_n*sizeof(int));
    cudaMemset(dp+(2*dp_n), 0, dp_n*sizeof(int));
    for(int len = 3; len <= dp_n; len++){
        int block_num = dp_n-len+1, num_data = len-2;
        max_reduce <<<block_num, NT, NT*sizeof(int)>>> (NUM, num_data, len, dp_n, dp);
    }
    cudaMemcpy(&res, dp+(dp_n*dp_n), sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dp);
}

__device__ int get_data(int left_idx, int len, int idx, int num_data, int n, int *dp){
    if(idx >= num_data) return 0;
    int right_idx = left_idx+len-1, left_len = idx+2, right_len = len-left_len+1, mid_idx = left_idx+left_len-1;
    return dp[left_len*n+left_idx]+dp[right_len*n+mid_idx]+data_GPU[left_idx]*data_GPU[mid_idx]*data_GPU[right_idx];
}

__global__ void max_reduce(int NUM, int num_data, int len, int n, int *dp){
    extern __shared__ int sdata[];
    int left_idx = blockIdx.x, tid = threadIdx.x, val = 0;
    for(int i = 0; i < NUM; i++){
        val = max(val, get_data(left_idx, len, i*(NT*2)+tid, num_data, n, dp));
        val = max(val, get_data(left_idx, len, i*(NT*2)+tid+NT, num_data, n, dp));
    }
    sdata[tid] = val;
    __syncthreads();
    for(int s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s) sdata[tid] = max(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }
    if(tid == 0) dp[len*n+left_idx] = sdata[0];
}