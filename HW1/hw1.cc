#include <iostream>
#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

void merge(float *data, float *tmp, int len, float *buf, int len_p, bool small){
    if(small){
        // if(len_p==0 || len==0 || data[len-1]<buf[0]) return;
        int data_idx = 0, buf_idx = 0;
        for(int j = 0; j < len; j++){
            if(data_idx<len && (buf_idx>=len_p || data[data_idx]<buf[buf_idx])) 
                tmp[j] = data[data_idx++];
            else 
                tmp[j] = buf[buf_idx++];
        }
    }
    else{
        // if(len_p==0 || len==0 || data[0]>buf[len_p-1]) return;
        int data_idx = len-1, buf_idx = len_p-1;
        for(int j = len-1; j >= 0; j--){
            if(data_idx>=0 && (buf_idx<0 || data[data_idx]>buf[buf_idx])) 
                tmp[j] = data[data_idx--];
            else 
                tmp[j] = buf[buf_idx--];
        }
    }
    // memcpy(data, tmp, len*sizeof(float));
}
int main(int argc, char **argv){
    // Input 
    long long n = atoll(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    // MPI Init
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Data size and start
    int unit = n/size, remain = n%size;
    int start = unit*rank+std::min(remain, rank);
    int len = unit+(rank < remain);

    float *data = new float[len];
    float *tmp = new float[len];
    float *buf = new float[unit+1];

    // 0 means even phase, 1 means odd phase
    int rank_p[2], len_p[2];
    if(rank & 1){
        rank_p[0] = rank==0?MPI_PROC_NULL:rank-1;
        rank_p[1] = rank==size-1?MPI_PROC_NULL:rank+1;
        len_p[0] = rank==0?0:(unit+(rank_p[0] < remain));
        len_p[1] = rank==size-1?0:(unit+(rank_p[1] < remain));
    }
    else{
        rank_p[0] = rank==size-1?MPI_PROC_NULL:rank+1;
        rank_p[1] = rank==0?MPI_PROC_NULL:rank-1;
        len_p[0] = rank==size-1?0:(unit+(rank_p[0] < remain));
        len_p[1] = rank==0?0:(unit+(rank_p[1] < remain));
    }

    // printf("%d %d %d %d %d\n", rank, even_rank, even_len, odd_rank, odd_len);

    // File read
    MPI_File input_file, output_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float)*start, data, len, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    // Sort local data
    boost::sort::spreadsort::spreadsort(data, data+len);
    // std::sort(data, data+len);

    // Even odd sort
    for(int i = 0; i < size+1; i += 2){
        // Even
        MPI_Sendrecv(data, len, MPI_FLOAT, rank_p[0], 0, buf, len_p[0], \
                    MPI_FLOAT, rank_p[0], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        merge(data, tmp, len, buf, len_p[0], rank<rank_p[0]);
        std::swap(data, tmp);
        // Odd
        MPI_Sendrecv(data, len, MPI_FLOAT, rank_p[1], 0, buf, len_p[1], \
                    MPI_FLOAT, rank_p[1], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        merge(data, tmp, len, buf, len_p[1], rank<rank_p[1]);
        std::swap(data, tmp);
    }
    // File write
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float)*start, data, len, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();

    delete[] data;
    delete[] tmp;
    delete[] buf;

    return 0;
}
