#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include <mpi.h>

// Global variavle
const char *job_name, *input_filename, *locality_config_filename, *output_dir;
int cpu_num, rank, mpi_num, num_reducer, delay, chunk_size;
std::vector<std::string> tmp_filename;
std::vector<std::ofstream> tmp_file;
std::vector<pthread_mutex_t> tmp_file_lock;
int pair_num;
pthread_mutex_t pair_num_lock;

// Major function
void Jobtracker();
void Tasktracker();
void* ThreadJob(void *arg);
void* MapperThread(void *arg);
void* ReducerThread(void *arg);
std::vector<std::pair<int, std::string>> InputSplit(std::vector<std::string>& input, int start);
std::unordered_map<std::string, int> Map(std::string& s);
int Partition(const std::string& s);
void Shuffle();
std::vector<std::pair<std::string, int>> ReadData(int id);
bool SortComparator(const std::pair<std::string, int>& a, const std::pair<std::string, int>& b);
std::vector<std::pair<std::string, std::vector<int>>> Group(std::vector<std::pair<std::string, int>>& data);
bool GroupComparator(const std::string& a, const std::string& b);
std::vector<std::pair<std::string, int>> Reduce(std::vector<std::pair<std::string, std::vector<int>>>& data);
void Output(std::vector<std::pair<std::string, int>>& data, int reducerId);

// Utility function
void Log(std::ofstream& file, std::string event, int taskID=-1, int nodeID=-1, unsigned long long start=-1, int num=-1);

// Thread argument
struct MapperArg{
    int threadId;
    std::vector<std::pair<int, std::string>> *record;
};
struct ReducerArg{
    int reducerId;
    std::vector<std::pair<std::string, int>> *data;
};

// For group
class DisjointSet{
private:
    std::unordered_map<std::string, std::string> parent;
    std::function<bool (const std::pair<std::string, int>&, const std::pair<std::string, int>&)> sortComparator;
public:
    DisjointSet(std::function<bool (const std::pair<std::string, int>&, const std::pair<std::string, int>&)> func): 
        sortComparator(func){}
    void set(std::string& s){
        parent[s] = s;
    }
    std::string find(std::string& s){
        if(parent[s] != s) parent[s] = find(parent[s]);
        return parent[s];
    }
    void join(std::string& s1, std::string& s2){
        std::string p1 = find(s1), p2 = find(s2);
        if(sortComparator({p1, 0}, {p2, 0})) parent[p2] = p1;
        else parent[p1] = p2;
    }
};

int main(int argc, char **argv){
    // Get number of CPU
    cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	cpu_num = CPU_COUNT(&cpuset);

    // Argument parsing
    assert(argc == 8);
    job_name = argv[1];
    num_reducer = strtol(argv[2], 0, 10);
    delay = strtol(argv[3], 0, 10);
    input_filename = argv[4];
    chunk_size = strtol(argv[5], 0, 10);
    locality_config_filename = argv[6];
    output_dir = argv[7];

    // MPI init
    MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_num);

    // Job and Task
    if(rank == 0) Jobtracker();
    else Tasktracker();

    // MPI Finalize
    MPI_Finalize();
    return 0;
}
void Jobtracker(){
    unsigned long long start = time(nullptr);
    // Open log file
    std::string log_filename = std::string(output_dir)+"/"+std::string(job_name)+"-log.out";
    std::ofstream log_file(log_filename, std::ofstream::out|std::ofstream::trunc);
    Log(log_file, "Start_Job");

    // Read locality config file
    std::ifstream locality_config_file(locality_config_filename);
    int chunkID, nodeID, worker_num = mpi_num-1, chunk_num = 0, buf;
    std::vector<std::queue<int>> V(worker_num+1);
    std::queue<int> Q;
    while(locality_config_file >> chunkID >> nodeID){
        nodeID = (nodeID+worker_num-1)%worker_num+1;
        V[nodeID].push(chunkID);
        Q.push(chunkID);
        chunk_num++;
    }
    locality_config_file.close();

    // Receive mapper requests from nodes
    std::vector<bool> used(chunk_num+1);
    std::vector<unsigned long long> start_time(chunk_num+1);
    int remain_chunk = chunk_num;
    MPI_Status status;
    while(remain_chunk--){
        MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(buf != -1) Log(log_file, "Complete_MapTask", buf, -1, start_time[buf]);
        nodeID = status.MPI_SOURCE;
        // Get chunk
        while(!V[nodeID].empty() && used[V[nodeID].front()]) V[nodeID].pop();
        // With data locality
        if(!V[nodeID].empty()){
            chunkID = V[nodeID].front(), V[nodeID].pop();
            used[chunkID] = true;
            start_time[chunkID] = time(nullptr);
            Log(log_file, "Dispatch_MapTask", chunkID, nodeID);
            MPI_Send(&chunkID, 1, MPI_INT, nodeID, 0, MPI_COMM_WORLD);
        }
        // Without data locality
        else{
            while(used[Q.front()]) Q.pop();
            chunkID = Q.front(), Q.pop();
            used[chunkID] = true;
            start_time[chunkID] = time(nullptr);
            Log(log_file, "Dispatch_MapTask", chunkID, nodeID);
            sleep(delay);
            MPI_Send(&chunkID, 1, MPI_INT, nodeID, 0, MPI_COMM_WORLD);
        }
    }

    // Send mapper task terminate flag
    int remain_worker = worker_num;
    while(remain_worker--){
        MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(buf != -1) Log(log_file, "Complete_MapTask", buf, -1, start_time[buf]);
        nodeID = status.MPI_SOURCE, buf = -1;
        MPI_Send(&buf, 1, MPI_INT, nodeID, 0, MPI_COMM_WORLD);
    }

    // Sum total pair_num
    buf = 0;
    MPI_Reduce(&buf, &pair_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Start shuffle
    unsigned long long start_shuffle = time(nullptr);
    Log(log_file, "Start_Shuffle", -1, -1, -1, pair_num);
    Shuffle();
    Log(log_file, "Finish_Shuffle", -1, -1, start_shuffle);

    // Receive reducer requests from nodes
    start_time.resize(num_reducer);
    for(int i = 0; i < num_reducer; i++){
        MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(buf != -1) Log(log_file, "Complete_ReduceTask", buf, -1, start_time[buf]);
        nodeID = status.MPI_SOURCE;
        start_time[i] = time(nullptr);
        Log(log_file, "Dispatch_ReduceTask", i, nodeID);
        MPI_Send(&i, 1, MPI_INT, nodeID, 0, MPI_COMM_WORLD);
    }
    // Send reducer task terminate flag
    remain_worker = worker_num;
    while(remain_worker--){
        MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(buf != -1) Log(log_file, "Complete_ReduceTask", buf, -1, start_time[buf]);
        nodeID = status.MPI_SOURCE, buf = -1;
        MPI_Send(&buf, 1, MPI_INT, nodeID, 0, MPI_COMM_WORLD);
    }
    
    // Close log file
    Log(log_file, "Finish_Job", -1, -1, start);
    log_file.close();
}
void Tasktracker(){
    // Read input file
    std::ifstream input_file(input_filename);
    std::string s;
    std::vector<std::string> input;
    while(std::getline(input_file, s))
        input.push_back(s);
    input_file.close();

    // Tmp_file init
    tmp_filename.resize(num_reducer);
    tmp_file.resize(num_reducer);
    tmp_file_lock.resize(num_reducer);
    for(int i = 0; i < num_reducer; i++){
        tmp_filename[i] = std::string(output_dir)+"/tmp_"+std::to_string(rank)+"_"+std::to_string(i)+".txt";
        tmp_file[i].open(tmp_filename[i], std::ofstream::out|std::ofstream::trunc);
        pthread_mutex_init(&tmp_file_lock[i], nullptr);
    }

    // Pair_num init
    pthread_mutex_init(&pair_num_lock, nullptr);

    // Mapper thread
    int buf = -1;
    MPI_Status status;
    pthread_t *mapper = new pthread_t[cpu_num-1];
    MapperArg *mapperArg = new MapperArg[cpu_num-1];
    while(true){
        MPI_Send(&buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&buf, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(buf == -1) break;
        int line_start = (buf-1)*chunk_size;
        auto record_list = InputSplit(input, line_start);
        for(int i = 0; i < cpu_num-1; i++){
            mapperArg[i].threadId = i;
            mapperArg[i].record = &record_list;
            pthread_create(&mapper[i], nullptr, &ThreadJob, static_cast<void*>(&mapperArg[i]));
        }
        for(int i = 0; i < cpu_num-1; i++){
            pthread_join(mapper[i], nullptr);
        }
    }
    delete mapper;
    delete mapperArg;

    // Close tmp_file
    for(int i = 0; i < num_reducer; i++){
        tmp_file[i].close();
        pthread_mutex_destroy(&tmp_file_lock[i]);
    }

    // Sum total pair_num
    MPI_Reduce(&pair_num, nullptr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    pthread_mutex_destroy(&pair_num_lock);

    // Reducer thread
    buf = -1;
    pthread_t *reducer = new pthread_t;
    ReducerArg *reducerArg = new ReducerArg;
    while(true){
        MPI_Send(&buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&buf, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(buf == -1) break;
        auto data = ReadData(buf);
        reducerArg->reducerId = buf;
        reducerArg->data = &data;
        pthread_create(reducer, nullptr, &ReducerThread, static_cast<void*>(reducerArg));
        pthread_join(*reducer, nullptr);
    }
    delete reducer;
    delete reducerArg;
}
void* ThreadJob(void *arg){
    MapperArg A = *static_cast<MapperArg*>(arg);
    for(int i = A.threadId; i < chunk_size; i += cpu_num-1)
        MapperThread(&((*A.record)[i]));
    return nullptr;
}
void* MapperThread(void *arg){
    auto [line_id, line_text] = *static_cast<std::pair<int, std::string>*>(arg);
    // Map function
    auto mp = Map(line_text);
    pthread_mutex_lock(&pair_num_lock);
    pair_num += mp.size();
    pthread_mutex_unlock(&pair_num_lock);
    // Partition fucntion
    std::vector<std::vector<std::pair<std::string, int>>> V(num_reducer);
    for(auto& [str, cnt] : mp){
        int reducer_id = Partition(str);
        V[reducer_id].push_back({str, cnt});
    }
    // Store data to tmp_file
    for(int i = 0; i < num_reducer; i++){
        pthread_mutex_lock(&tmp_file_lock[i]);
        for(auto& [str, cnt] : V[i])
            tmp_file[i] << str << " " << cnt << "\n";
        pthread_mutex_unlock(&tmp_file_lock[i]);
    }
    return nullptr;
}
void* ReducerThread(void *arg){
    ReducerArg A = *static_cast<ReducerArg*>(arg);
    auto data = *A.data;
    // Sort function
    std::sort(data.begin(), data.end(), SortComparator);
    // Group function
    auto group = Group(data);
    // Reduce function
    auto reduce = Reduce(group);
    // Output function
    Output(reduce, A.reducerId);
    return nullptr;
}
std::vector<std::pair<int, std::string>> InputSplit(std::vector<std::string>& input, int start){
    std::vector<std::pair<int, std::string>> res(chunk_size);
    for(int i = 0; i < chunk_size; i++){
        res[i].first = start+i+1;
        res[i].second = input[start+i];
    }
    return res;
}
std::unordered_map<std::string, int> Map(std::string& s){
    s += " ";
    std::unordered_map<std::string, int> res;
    int start = 0;
    for(int i = 0; i < s.size(); i++){
        if(isalpha(s[i])) continue;
        if(i != start) res[s.substr(start, i-start)]++;
        start = i+1;
    }
    return res;
}
int Partition(const std::string& s){
    return std::hash<std::string>{}(s)%num_reducer;
}
void Shuffle(){
    for(int i = 0; i < num_reducer; i++){
        std::string out_filename = std::string(output_dir)+"/tmp_"+std::to_string(i)+".txt";
        std::ofstream output_file(out_filename, std::ofstream::out|std::ofstream::trunc);
        for(int j = 1; j < mpi_num; j++){
            std::string in_filename = std::string(output_dir)+"/tmp_"+std::to_string(j)+"_"+std::to_string(i)+".txt";
            std::ifstream input_file(in_filename);
            std::string s;
            while(getline(input_file, s))
                output_file << s << "\n";
            input_file.close();
            remove(in_filename.c_str());
        }
        output_file.close();
    }
}
std::vector<std::pair<std::string, int>> ReadData(int id){
    std::string in_filename = std::string(output_dir)+"/tmp_"+std::to_string(id)+".txt";
    std::ifstream input_file(in_filename);
    std::vector<std::pair<std::string, int>> res;
    int cnt;
    std::string str;
    while(input_file >> str >> cnt)
        res.push_back({str, cnt});
    input_file.close();
    remove(in_filename.c_str());
    return res;
}
bool SortComparator(const std::pair<std::string, int>& a, const std::pair<std::string, int>& b){
    return a.first < b.first;
}
std::vector<std::pair<std::string, std::vector<int>>> Group(std::vector<std::pair<std::string, int>>& data){
    int n = data.size();
    DisjointSet ds(SortComparator);
    for(int i = 0; i < n; i++) ds.set(data[i].first);
    for(int i = 0; i < n; i++)
        for(int j = i+1; j < n; j++)
            if(GroupComparator(data[i].first, data[j].first)) ds.join(data[i].first, data[j].first);
    std::vector<std::pair<std::string, std::vector<int>>> res;
    std::unordered_map<std::string, int> seen;
    for(auto& [str, cnt] : data){
        std::string p = ds.find(str);
        if(!seen.count(p)){
            seen[p] = res.size();
            res.push_back({p, std::vector<int>{cnt}});
        }
        else res[seen[p]].second.push_back(cnt);
    }
    return res;
}
bool GroupComparator(const std::string& a, const std::string& b){
    return a == b;
}
std::vector<std::pair<std::string, int>> Reduce(std::vector<std::pair<std::string, std::vector<int>>>& data){
    int n = data.size();
    std::vector<std::pair<std::string, int>> res(n);
    for(int i = 0; i < n; i++){
        res[i].first = data[i].first;
        res[i].second = std::accumulate(data[i].second.begin(), data[i].second.end(), 0);
    }
    return res;
}
void Output(std::vector<std::pair<std::string, int>>& data, int reducerId){
    std::string out_filename = std::string(output_dir)+"/"+std::string(job_name)+"-"+std::to_string(reducerId)+".out";
    std::ofstream output_file(out_filename, std::ofstream::out|std::ofstream::trunc);
    for(auto& [str, cnt] : data)
        output_file << str << " " << cnt << "\n";
    output_file.close();
}
void Log(std::ofstream& file, std::string event, int taskID, int nodeID, unsigned long long start, int num){
    unsigned long long t = time(nullptr);
    if(event == "Start_Job"){
        file << t << "," << event << "," << job_name << "," << mpi_num << "," << cpu_num << "," << num_reducer << "," << \
            delay << "," << input_filename << "," << chunk_size << "," << locality_config_filename << "," << output_dir << "\n";
    }
    else if(event=="Dispatch_MapTask" || event=="Dispatch_ReduceTask"){
        file << t << "," << event << "," << taskID << "," << nodeID << "\n";
    }
    else if(event=="Complete_MapTask" || event=="Complete_ReduceTask"){
        int exec_time = t-start;
        file << t << "," << event << "," << taskID << "," << exec_time << "\n";
    }
    else if(event == "Start_Shuffle"){
        file << t << "," << event << "," << num << "\n";
    }
    else if(event=="Finish_Shuffle" || event=="Finish_Job"){
        int exec_time = t-start;
        file << t << "," << event << "," << exec_time << "\n";
    }
}