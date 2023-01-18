#include <iostream>
#include <fstream>
#include <random>
#include <cstdlib>
#define N 10000
#define MIN 0
#define MAX 10
const char *filename;
int n, data[N+1];
int main(int argc, char* argv[]){
    filename = argv[1];
    n = strtol(argv[2], 0, 10);
    if(n<=0 || n>N){
        std::cout << "Invalid n\n";
        return 0; 
    }

    // random setup
    std::uniform_int_distribution<std::mt19937::result_type> udist(MIN, MAX);
    std::mt19937 rng;
    std::mt19937::result_type const seedval = 0;
    rng.seed(seedval);

    data[0] = n;
    for(int i = 1; i <= n; i++)
        data[i] = udist(rng);

    // write output file
    std::ofstream outfile(filename, std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(data), (n+1)*sizeof(int));
    outfile.close();
    
    return 0;
}