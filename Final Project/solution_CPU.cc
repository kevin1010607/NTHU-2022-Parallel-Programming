#include <fstream>
#include <sys/time.h>
#define US_PER_SEC 1000000
#define N 5000
const char *input_filename, *output_filename;
int n, new_n, data[N+2], dp[N+2][N+2];

void input();
void output();
void solve();

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
    outfile.write(reinterpret_cast<const char*>(&dp[0][new_n-1]), sizeof(int));
    outfile.close();
}

void solve(){
    // solution
    data[0] = data[n+1] = 1;
    new_n = n+2;
    for(int len = 3; len <= new_n; len++){
        for(int i = 0; i < new_n-len+1; i++){
            int j = i+len-1;
            for(int k = i+1; k < j; k++)
                dp[i][j] = std::max(dp[i][j], dp[i][k]+dp[k][j]+data[i]*data[k]*data[j]);
        }
    }
}