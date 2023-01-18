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
__global__ void max_reduce(int, int, int*);

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
    cudaMemcpyToSymbol(data_GPU, data, dp_n*sizeof(int));
    cudaMalloc(&dp, dp_n*dp_n*sizeof(int));
    cudaMemset(dp, 0, dp_n*dp_n*sizeof(int));
    for(int len = 3; len <= dp_n; len++){
        int num = dp_n-len+1, block_num = (num+NT-1)/NT;
        max_reduce <<<block_num, NT>>> (len, dp_n, dp);
    }
    cudaMemcpy(&res, dp+(dp_n-1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dp);
}

__global__ void max_reduce(int len, int n, int *dp){
    int i = blockIdx.x*blockDim.x+threadIdx.x, j = i+len-1;
    if(i >= n-len+1) return;
    for(int k = i+1; k < j; k++)
        dp[i*n+j] = max(dp[i*n+j], dp[i*n+k]+dp[k*n+j]+data_GPU[i]*data_GPU[k]*data_GPU[j]);
}