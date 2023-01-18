#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <map>
#include <algorithm>
using namespace std;
map<string, int> Count(string filename){
    map<string, int> res;
    string str;
    int cnt;
    ifstream file(filename);
    while(file >> str >> cnt)
        res[str] += cnt;
    return res;
}
bool Compare(map<string, int>& m1, map<string, int>& m2){
    return m1.size()==m2.size() && equal(m1.begin(), m1.end(), m2.begin());
}
int main(int argc, char **argv){
    // Argument parsing
    assert(argc == 5);
    string job_name = string(argv[1]);
    int num_reducer = strtol(argv[2], 0, 10);
    string input_filename = string(argv[3]);
    string output_dir = string(argv[4]);

    // Count
    map<string, int> answer = Count(input_filename);
    map<string, int> myanswer;
    for(int i = 0; i < num_reducer; i++){
        string filename = output_dir+"/"+job_name+"-"+to_string(i)+".out";
        for(auto& [str, cnt] : Count(filename))
            myanswer[str] += cnt;
    }

    // Compare
    cout << "########## " << (Compare(answer, myanswer)?"Accept":"Error") << " ##########\n";
    return 0;
}