#include <fstream>
#include <cuda.h>
#include <sys/time.h>
#define US_PER_SEC 1000000
#define N 10000
#define NT 1024
// CPU
const char *input_filename, *output_filename;
int n, dp_n, reduce_n, data[N+2], res;
// GPU
__constant__ int data_GPU[N+2];
int *dp, *reduce_data;

// CPU
void input();
void output();
void solve();
// GPU
__global__ void read_input(int, int, int*, int, int*);
__global__ void write_output(int, int, int*, int, int*);
__global__ void max_reduce(int, int*);

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
    reduce_n = (n+NT-1)/NT*NT;
    cudaMemcpyToSymbol(data_GPU, data, dp_n*sizeof(int));
    cudaMalloc(&dp, (dp_n+1)*dp_n*sizeof(int));
    cudaMemset(dp+(2*dp_n), 0, dp_n*sizeof(int));
    cudaMalloc(&reduce_data, dp_n*reduce_n*sizeof(int));
    for(int len = 3; len <= dp_n; len++){
        int blockX_num = dp_n-len+1, num_data = len-2;
        read_input <<<blockX_num, NT>>> (len, dp_n, dp, reduce_n, reduce_data);
        while(num_data > 1){
            int blockY_num = (num_data+NT-1)/NT;
            max_reduce <<<dim3(blockX_num, blockY_num), NT, NT*sizeof(int)>>> (reduce_n, reduce_data);
            num_data = blockY_num;
        }
        write_output <<<blockX_num, 1>>> (len, dp_n, dp, reduce_n, reduce_data);
    }
    cudaMemcpy(&res, dp+(dp_n*dp_n), sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dp);
    cudaFree(reduce_data);
}

__global__ void read_input(int len, int dp_n, int *dp, int reduce_n, int *reduce_data){
    int left_idx = blockIdx.x, right_idx = blockIdx.x+len-1, tid = threadIdx.x;
    for(int i = tid; i < len-2; i += blockDim.x){
        int left_len = i+2, right_len = len-left_len+1, mid_idx = left_idx+left_len-1;
        reduce_data[left_idx*reduce_n+i] = dp[left_len*dp_n+left_idx]+dp[right_len*dp_n+mid_idx]+ \
            data_GPU[left_idx]*data_GPU[mid_idx]*data_GPU[right_idx];
    }
}

__global__ void write_output(int len, int dp_n, int *dp, int reduce_n, int *reduce_data){
    int left_idx = blockIdx.x;
    dp[len*dp_n+left_idx] = reduce_data[left_idx*reduce_n];
}

__global__ void max_reduce(int n, int *reduce_data){
    extern __shared__ int sdata[];
    int left_idx = blockIdx.x, data_idx = blockIdx.y*blockDim.x+threadIdx.x, tid = threadIdx.x;
    sdata[tid] = reduce_data[left_idx*n+data_idx];
    __syncthreads();
    for(int s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s) sdata[tid] = max(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }
    if(tid == 0) reduce_data[left_idx*n+blockIdx.y] = sdata[0];
}