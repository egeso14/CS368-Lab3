#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

void allocate_memory_on_device_32(uint32_t* address, size_t size)
{
    cudamalloc(&address, size);
}

void allocate_memory_on_device_8(uint8_t* address, size_t size)
{
    cudamalloc(&address, size);
}

void copy_memory_to_device_32(uint32_t* devicePtr, uint32_t* hostPtr, size_t size)
{
    cudamemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
}

void copy_memory_to_device_8(uint8_t* devicePtr, uint8_t* hostPtr, size_t size)
{
    cudamemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
}


void opt_2dhisto( uint32_t** cont_input, uint8_t* bins, int dataN, int width_input, int height_input, int dim_grid_x, int dim_grid_y, int blockX, int blockY)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    dim3 dim_block(blockX, blockY);
    dim3 dim_grid(dim_grid_x, dim_grid_y); // block goes down 128 times?


    opt_2d_histo_kernel<<<dim_grid, dim_block>>>(cont_input, bins, dataN, width_input, height_input);
}


__constant__ int BIN_COUNT = 1024; // number of bins in sub-histogram
/* Include below the implementation of any other functions you need */

__global__ void opt_2dhisto_kernel(uint32_t* cont_input, uint8_t* bins, int width_input, int height_input)
{
    // Width of image (3084 (or 4096 padded) broken down into multiple tiles of a row split into various blocks. 
    // Currently static with 16 threads per block
    
    
    const int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int numThreads = blockDim.x * gridDim.x * blockDim.y;

    __shared__ uint s_Hist[BIN_COUNT];

    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x)
        s_Hist[i] = 0;
    __syncthreads();



    for (int changed_y = globalY; changed_y < height_input; changed_y += blockDim.y)
    {
        if (globalX < 3984) 
        {
            uint64_t index = changed_y * width_input + globalX;
            uint32_t data4 = cont_input[index]; //contains 4 pievces of data, access is coalesced
            
            atomicAdd(&s_Hist[(data4 >>  0) & 0xFFU], 1);
            atomicAdd(&s_Hist[(data4 >>  8) & 0xFFU], 1);
            atomicAdd(&s_Hist[(data4 >> 16) & 0xFFU], 1);
            atomicAdd(&s_Hist[(data4 >> 24) & 0xFFU], 1);
        }

    }

    __syncthreads();
    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x)  atomicAdd(&bins[i], s_Hist[i]);
    
       
    
}