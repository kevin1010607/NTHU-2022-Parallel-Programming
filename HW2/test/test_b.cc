#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <emmintrin.h>
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))
#define ll long long

typedef struct{
    ll F, S;
}Pair;

int *image, width, height, iters, cpu_num;
ll idx, total, CHUNK, unit, remain, start, len;
double left, right, lower, upper;

omp_lock_t lock;

Pair get_chunk(){
    omp_set_lock(&lock);
    Pair res = {idx, min(CHUNK, start+len-idx)};
    idx += res.S;
    omp_unset_lock(&lock);
    return res;
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    int size, rank;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    double begin, end;
    begin = MPI_Wtime();

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 10);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    cpu_num = strtod(argv[9], 0);
    total = (ll)width*height;

    unit = total/size, remain = total%size;
    start = unit*rank+min(remain, rank);
    len = unit+(rank < remain);

    idx = start;
    CHUNK = max(2, len/cpu_num/30);

    /* allocate memory for image */
    image = (int*)malloc((rank==0?total:len) * sizeof(int));
    assert(image);

    omp_init_lock(&lock);

    /* mandelbrot set */
#pragma omp parallel num_threads(cpu_num)
{
    int tid = omp_get_thread_num();
    double _begin, _end;
    _begin = omp_get_wtime();

    __m128d SIMD_2 = _mm_set1_pd(2);    
    while(true){
        Pair p = get_chunk();
        if(p.S == 0) break;

        ll idx1 = p.F, idx2 = p.F+1;
        bool end1 = false, end2 = (idx2==p.F+p.S);
        if(end2) idx2 = p.F+p.S-1;

        int j1 = idx1/width, i1 = idx1%width;
        int j2 = idx2/width, i2 = idx2%width;
        double y0_1 = j1 * ((upper - lower) / height) + lower;
        double x0_1 = i1 * ((right - left) / width) + left;
        double y0_2 = j2 * ((upper - lower) / height) + lower;
        double x0_2 = i2 * ((right - left) / width) + left;

        int repeats1 = 0, repeats2 = 0;
        double tmp;

        __m128d SIMD_y0 = _mm_set_pd(y0_1, y0_2);
        __m128d SIMD_x0 = _mm_set_pd(x0_1, x0_2);
        __m128d SIMD_x = _mm_set1_pd(0);
        __m128d SIMD_y = _mm_set1_pd(0);
        __m128d SIMD_squared = _mm_set1_pd(0);

        while(!end1 || !end2){
            __m128d SIMD_tmp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(SIMD_x, SIMD_x), _mm_mul_pd(SIMD_y, SIMD_y)), SIMD_x0);
            SIMD_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(SIMD_x, SIMD_y), SIMD_2), SIMD_y0);
            SIMD_x = SIMD_tmp;
            SIMD_squared = _mm_add_pd(_mm_mul_pd(SIMD_x, SIMD_x), _mm_mul_pd(SIMD_y, SIMD_y));
            if(!end1){
                _mm_storeh_pd(&tmp, SIMD_squared);
                if(++repeats1==iters || tmp>=4){
                    image[j1 * width + i1 - start] = repeats1;
                    if(max(idx1, idx2)+1 != p.F+p.S){
                        idx1 = max(idx1, idx2)+1;
                        j1 = idx1/width, i1 = idx1%width;
                        y0_1 = j1 * ((upper - lower) / height) + lower;
                        x0_1 = i1 * ((right - left) / width) + left;
                        SIMD_y0 = _mm_set_pd(y0_1, y0_2);
                        SIMD_x0 = _mm_set_pd(x0_1, x0_2);
                        repeats1 = 0;

                        _mm_storel_pd(&tmp, SIMD_x);
                        SIMD_x = _mm_set_pd(0, tmp);
                        _mm_storel_pd(&tmp, SIMD_y);
                        SIMD_y = _mm_set_pd(0, tmp);
                        _mm_storel_pd(&tmp, SIMD_squared);
                        SIMD_squared = _mm_set_pd(0, tmp);
                    }
                    else end1 = true, idx1 = p.F+p.S-1;
                }
            }
            if(!end2){
                _mm_storel_pd(&tmp, SIMD_squared);
                if(++repeats2==iters || tmp>=4){
                    image[j2 * width + i2 - start] = repeats2;
                    if(max(idx1, idx2)+1 != p.F+p.S){
                        idx2 = max(idx1, idx2)+1;
                        j2 = idx2/width, i2 = idx2%width;
                        y0_2 = j2 * ((upper - lower) / height) + lower;
                        x0_2 = i2 * ((right - left) / width) + left;
                        SIMD_y0 = _mm_set_pd(y0_1, y0_2);
                        SIMD_x0 = _mm_set_pd(x0_1, x0_2);
                        repeats2 = 0;

                        _mm_storeh_pd(&tmp, SIMD_x);
                        SIMD_x = _mm_set_pd(tmp, 0);
                        _mm_storeh_pd(&tmp, SIMD_y);
                        SIMD_y = _mm_set_pd(tmp, 0);
                        _mm_storeh_pd(&tmp, SIMD_squared);
                        SIMD_squared = _mm_set_pd(tmp, 0);
                    }
                    else end2 = true, idx2 = p.F+p.S-1;
                }
            }
        }
    }

    _end = omp_get_wtime();
    printf("%lf %d\n", _end-_begin, tid, rank);
}
    omp_destroy_lock(&lock);

    end = MPI_Wtime();
    printf("%lf\n", end-begin);

    /* draw and cleanup */
    if(rank == 0){
        for(int i = 1; i < size; i++){
            int t = unit+(i < remain);
            MPI_Recv(image+len, t, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            len += t;
        }
        write_png(filename, iters, width, height, image);
    }
    else{
        MPI_Send(image, len, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    free(image);
}
