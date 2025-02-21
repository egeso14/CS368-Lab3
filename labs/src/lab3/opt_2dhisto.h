#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t** cont_input,
    uint8_t* bins,
    int dataN,
    int width_input,
    int height_input,
    int dim_grid,
    int dim_block);

/* Include below the function headers of any other functions that you implement */
#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <stdint.h>
#include <stdlib.h>




void allocate_memory_on_device_32(uint32_t* address, size_t size);

void allocate_memory_on_device_8(uint8_t* address, size_t size);

void copy_memory_to_device_32(uint32_t* devicePtr, uint32_t* hostPtr, size_t size);


void copy_memory_to_device_8(uint8_t* devicePtr, uint8_t* hostPtr, size_t size);

// (Optional) Kernel declaration if needed externally.
__global__ void opt_2dhisto_kernel(uint32_t* cont_input,
                                   uint8_t* bins,
                                   int dataN,
                                   int width_input,
                                   int height_input);


#endif // CUDA_UTIL_H

#endif
