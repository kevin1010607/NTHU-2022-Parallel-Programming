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
#include <pthread.h>
#include <emmintrin.h>
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))
#define ll long long

typedef struct{
    ll F, S;
}Pair;

int *image, width, height, iters, cpu_num;
ll idx, total, CHUNK;
double left, right, lower, upper;

pthread_mutex_t mutex;

Pair get_chunk(){
    pthread_mutex_lock(&mutex);
    Pair res = {idx, min(CHUNK, total-idx)};
    idx += res.S;
    pthread_mutex_unlock(&mutex);
    return res;
}

void* solve(void *attr){
    int id = *(int*)attr;
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
                    image[j1 * width + i1] = repeats1;
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
                    image[j2 * width + i2] = repeats2;
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
    pthread_exit(NULL);
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
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);
    pthread_t T[12];
	int id[12];

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    total = (ll)width*height;
    CHUNK = max(2, total/cpu_num/50);

    /* allocate memory for image */
    image = (int*)malloc(total * sizeof(int));
    assert(image);

    pthread_mutex_init(&mutex, 0);

    /* mandelbrot set */
    for(int i = 0; i < cpu_num; i++){
		id[i] = i;
		pthread_create(&T[i], NULL, solve, (void*)&id[i]);
	}

	for(int i = 0; i < cpu_num; i++){
		pthread_join(T[i], NULL);
    }

    pthread_mutex_destroy(&mutex);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
