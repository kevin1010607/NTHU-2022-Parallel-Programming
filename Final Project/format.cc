#include <iostream>
#include <fstream>
#define N 10000
const char *filename;
int n, data[N];
int main(int argc, char* argv[]){
    filename = argv[1];

    // read input file
    std::ifstream infile(filename, std::ios::binary);
    infile.read(reinterpret_cast<char*>(&n), sizeof(int));
    infile.read(reinterpret_cast<char*>(data), n*sizeof(int));
    infile.close();
    
    // write output
    std::cout << "[";
    for(int i = 0; i < n; i++)
        std::cout << data[i] << ",]"[i==n-1];
    std::cout << "\n";
    
    return 0;
}