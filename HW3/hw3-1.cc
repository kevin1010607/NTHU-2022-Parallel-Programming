#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <emmintrin.h>
#include <smmintrin.h>

const int INF = 1073741823;
const int V = 6010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int min(int a, int b);
int max(int a, int b);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m, cpu_num;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);

    input(argv[1]);
    int B = 64;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; j++){
            Dist[i][j] = i==j?0:INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int min(int a, int b) { return a<b?a:b; }
int max(int a, int b) { return a>b?a:b; }
int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);
        /* Phase 2*/
        cal(B, r, r, 0, round, 1);
        cal(B, r, 0, r, 1, round);
        /* Phase 3*/
        cal(B, r, 0, 0, round, round);
    }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;
    int k_start = Round*B, k_end = min((Round+1)*B, n);

    #pragma omp parallel for num_threads(cpu_num) schedule(dynamic)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        int block_internal_start_x = b_i*B, block_internal_end_x = min((b_i+1)*B, n);
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            int block_internal_start_y = b_j*B, block_internal_end_y = min((b_j+1)*B, n);
            for (int k = k_start; k < k_end; ++k) {
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    __m128i SIMD_ik = _mm_set1_epi32(Dist[i][k]);
                    for (int j = block_internal_start_y; j+3 < block_internal_end_y; j += 4) {
                        __m128i SIMD_l = _mm_loadu_si128((__m128i*)&Dist[i][j]);;
                        __m128i SIMD_r = _mm_add_epi32(SIMD_ik, _mm_loadu_si128((__m128i*)&Dist[k][j]));
                        _mm_storeu_si128((__m128i*)&Dist[i][j], _mm_min_epi32(SIMD_l, SIMD_r));
                    }
                    int j = block_internal_start_y+(block_internal_end_y-block_internal_start_y)/4*4;
                    while(j < block_internal_end_y){
                        Dist[i][j] = min(Dist[i][j], Dist[i][k]+Dist[k][j]);
                        j++;
                    }
                }
            }
        }
    }
}
