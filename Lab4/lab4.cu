#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define Z 2
#define Y 5
#define X 5
#define xB 2
#define yB 2
#define SCALE 8

#define T 256

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

// inline __device__ int bound_check(int val, int lower, int upper) {
//     if (val >= lower && val < upper)
//         return 1;
//     else
//         return 0;
// }

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= width) return;
    int x = tid, idx = threadIdx.x, now = yB;

    __shared__ char mask[Z][Y][X];

    mask[0][0][0] = -1; mask[0][0][1] = -4; mask[0][0][2] = -6; mask[0][0][3] = -4; mask[0][0][4] = -1;
    mask[0][1][0] = -2; mask[0][1][1] = -8; mask[0][1][2] = -12; mask[0][1][3] = -8; mask[0][1][4] = -2;
    mask[0][2][0] = 0; mask[0][2][1] = 0; mask[0][2][2] = 0; mask[0][2][3] = 0; mask[0][2][4] = 0;
    mask[0][3][0] = 2; mask[0][3][1] = 8; mask[0][3][2] = 12; mask[0][3][3] = 8; mask[0][3][4] = 2;
    mask[0][4][0] = 1; mask[0][4][1] = 4; mask[0][4][2] = 6; mask[0][4][3] = 4; mask[0][4][4] = 1;
    mask[1][0][0] = -1; mask[1][0][1] = -2; mask[1][0][2] = 0; mask[1][0][3] = 2; mask[1][0][4] = 1;
    mask[1][1][0] = -4; mask[1][1][1] = -8; mask[1][1][2] = 0; mask[1][1][3] = 8; mask[1][1][4] = 4;
    mask[1][2][0] = -6; mask[1][2][1] = -12; mask[1][2][2] = 0; mask[1][2][3] = 12; mask[1][2][4] = 6;
    mask[1][3][0] = -4; mask[1][3][1] = -8; mask[1][3][2] = 0; mask[1][3][3] = 8; mask[1][3][4] = 4;
    mask[1][4][0] = -1; mask[1][4][1] = -2; mask[1][4][2] = 0; mask[1][4][3] = 2; mask[1][4][4] = 1;

    __shared__ unsigned char ss[Y][T+xB+xB][3];
    for (int v = -yB; v <= yB; ++v) {
        for (int u = -xB; u <= xB; ++u) {
            int flag = x+u>=0 && x+u<width && v>=0 && v<height;
            ss[v+yB][u+(idx+xB)][2] = flag?s[channels*(width*(0+v)+(x+u))+2]:0;
            ss[v+yB][u+(idx+xB)][1] = flag?s[channels*(width*(0+v)+(x+u))+1]:0;
            ss[v+yB][u+(idx+xB)][0] = flag?s[channels*(width*(0+v)+(x+u))+0]:0;
        }
    }

    for (int y = 0; y < height; ++y) {
        /* Z axis of mask */
        float val_02 = 0., val_01 = 0., val_00 = 0.;
        float val_12 = 0., val_11 = 0., val_10 = 0.;

        /* Y and X axis of mask */
        __syncthreads();
        for (int v = -yB; v <= yB; ++v) {
            for (int u = -xB; u <= xB; ++u) {
                const unsigned char R = ss[(v+now+Y)%Y][u+(idx+xB)][2];
                const unsigned char G = ss[(v+now+Y)%Y][u+(idx+xB)][1];
                const unsigned char B = ss[(v+now+Y)%Y][u+(idx+xB)][0];

                val_02 += R * mask[0][u + xB][v + yB];
                val_01 += G * mask[0][u + xB][v + yB];
                val_00 += B * mask[0][u + xB][v + yB];

                val_12 += R * mask[1][u + xB][v + yB];
                val_11 += G * mask[1][u + xB][v + yB];
                val_10 += B * mask[1][u + xB][v + yB];
            }
        }

        for (int u = -xB; u <= xB; ++u) {
            int flag = x+u>=0 && x+u<width && y+1+yB>=0 && y+1+yB<height;
            ss[(-yB+now+Y)%Y][u+(idx+xB)][2] = flag?s[channels*(width*(y+1+yB)+(x+u))+2]:0;
            ss[(-yB+now+Y)%Y][u+(idx+xB)][1] = flag?s[channels*(width*(y+1+yB)+(x+u))+1]:0;
            ss[(-yB+now+Y)%Y][u+(idx+xB)][0] = flag?s[channels*(width*(y+1+yB)+(x+u))+0]:0;
        }
        now = (now+1)%Y;

        t[channels * (width * y + x) + 2] = min(255., sqrt(val_02*val_02+val_12*val_12)/SCALE);
        t[channels * (width * y + x) + 1] = min(255., sqrt(val_01*val_01+val_11*val_11)/SCALE);
        t[channels * (width * y + x) + 0] = min(255., sqrt(val_00*val_00+val_10*val_10)/SCALE);
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    // decide to use how many blocks and threads
    const int num_threads = T;
    const int num_blocks = width / num_threads + 1;

    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // launch cuda kernel
    sobel <<<num_blocks, num_threads>>> (dsrc, ddst, height, width, channels);

    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}

