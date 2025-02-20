#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

void opt_2dhisto( uint32_t** input, uint8_t* bins, int dataN, int width_input, int height_input, int dim_grid, int dim_block)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    MatrixMulKernel<<<dim_grid, dim_block>>>(input, bins, dataN, width_input, height_input);
}


__constant__ int BIN_COUNT = 1024; // number of bins in sub-histogram
/* Include below the implementation of any other functions you need */

__global__ void opt_2dhisto_kernel(uint32_t** input, uint8_t* bins, int dataN, int width_input, int height_input)
{
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int numThreads = blockDim.x * gridDim.x;

    __shared__ uint s_Hist[BIN_COUNT];

    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x)
        s_Hist[i] = 0;
    __syncthreads();

    for (int pos = globalTid; pos < dataN; pos += numThreads)
    {
        int inputIndexY = pos / width_input;
        int inputIndexX = pos % width_input;

        uint32_t data4 = input[inputIndexY][inputIndexX]; //contains 4 pievces of data, access is coalesced
        
        atomicAdd(&s_Hist[(data4 >>  0) & 0xFFU], 1);
        atomicAdd(&s_Hist[(data4 >>  8) & 0xFFU], 1);
        atomicAdd(&s_Hist[(data4 >> 16) & 0xFFU], 1);
        atomicAdd(&s_Hist[(data4 >> 24) & 0xFFU], 1);

    }

    __syncthreads();
    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x)  atomicAdd(&bins[i], s_Hist[i]);
    
       
    
}