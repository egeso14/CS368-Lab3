#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cutil.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

#define SQRT_2    1.4142135623730950488
#define SPREAD_BOTTOM   (2)
#define SPREAD_TOP      (6)

#define NEXT(init_, spread_)\
    (init_ + (int)((drand48() - 0.5) * (drand48() - 0.5) * 4.0 * SQRT_2 * SQRT_2 * spread_));

#define CLAMP(value_, min_, max_)\
    if (value_ < 0)\
        value_ = (min_);\
    else if (value_ > (max_))\
        value_ = (max_);

// Generate another bin for the histogram.  The bins are created as a random walk ...
static uint32_t next_bin(uint32_t pix)
{
    const uint16_t bottom = pix & ((1<<HISTO_LOG)-1);
    const uint16_t top   = (uint16_t)(pix >> HISTO_LOG);

    int new_bottom = NEXT(bottom, SPREAD_BOTTOM)
    CLAMP(new_bottom, 0, HISTO_WIDTH-1)

    int new_top = NEXT(top, SPREAD_TOP)
    CLAMP(new_top, 0, HISTO_HEIGHT-1)

    const uint32_t result = (new_bottom | (new_top << HISTO_LOG)); 

    return result; 
}

// Return a 2D array of histogram bin-ids.  This function generates
// bin-ids with correlation characteristics similar to some actual images.
// The key point here is that the pixels (and thus the bin-ids) are *NOT*
// randomly distributed ... a given pixel tends to be similar to the
// pixels near it.
static uint32_t **generate_histogram_bins()
{
    uint32_t **input = (uint32_t**)alloc_2d(INPUT_HEIGHT, INPUT_WIDTH, sizeof(uint32_t));

    input[0][0] = HISTO_WIDTH/2 | ((HISTO_HEIGHT/2) << HISTO_LOG);
    for (int i = 1; i < INPUT_WIDTH; ++i)
        input[0][i] =  next_bin(input[0][i - 1]);
    for (int j = 1; j < INPUT_HEIGHT; ++j)
    {
        input[j][0] =  next_bin(input[j - 1][0]);
        for (int i = 1; i < INPUT_WIDTH; ++i)
            input[j][i] =  next_bin(input[j][i - 1]);
    }

    return input;
}

int main(int argc, char* argv[])
{
    /* Case of 0 arguments: Default seed is used */
    if (argc < 2){
	srand48(0);
    }
    /* Case of 1 argument: Seed is specified as first command line argument */ 
    else {
	int seed = atoi(argv[1]);
	srand48(seed);
    }

    uint8_t *gold_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

    // Use kernel_bins for your final result
    uint8_t *kernel_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

    // A 2D array of histogram bin-ids.  One can think of each of these bins-ids as
    // being associated with a pixel in a 2D image.
    uint32_t **input = generate_histogram_bins();

    TIME_IT("ref_2dhisto",
            1000,
            ref_2dhisto(input, INPUT_HEIGHT, INPUT_WIDTH, gold_bins);)

    /* Include your setup code below (temp variables, function calls, etc.) */
    
    // do memory allocation

    int dataN = INPUT_HEIGHT * INPUT_WIDTH;
    
    uint8_t *d_kernel_bins;
    
    size_t input_width_padded_to_32 = (INPUT_WIDTH + 128) & ~127;
    uint32_t *input_start = input[0];
    uint32_t *d_input;

    size_t histogram_size = HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t);
    allocate_memory_on_device_8(d_kernel_bins, histogram_size);
    allocate_memory_on_device_32(d_input, input_width_padded_to_32*INPUT_HEIGHT*sizeof(uint32_t));

    // copy them over
    copy_memory_to_device_8(d_kernel_bins, kernel_bins, histogram_size);
    copy_memory_to_device_32(d_input, input_start, input_width_padded_to_32*INPUT_HEIGHT*sizeof(uint32_t));

    // setup execution parameters
    // we don't want to read padding, that means our block should have as many threads as will be enugh for INPUT_WİDTH

    int blockX = 32;  // try this setup first because it allows us to get rid of if statements in the code
    int blockY = 32;    // no coalescing unless each block does multiple reads because of block dim
                    // but we also don't want 200-something blocks
                    // how many boxes should each thread cover?

    dim_grid_x = 128;
    dim_grid_y = 128;



    
   
    

    opt_2dhisto(d_input, d_kernel_bins, input_width_padded_to_32, INPUT_HEIGHT, dim_grid_x, dim_grid_y, blockX, blockY);
    

    /* End of setup code */

    /* This is the call you will use to time your parallel implementation */
    TIME_IT("opt_2dhisto",
            1000,
            opt_2dhisto( /*Define your own function parameters*/ );)

    /* Include your teardown code below (temporary variables, function calls, etc.) */



    /* End of teardown code */

    int passed=1;
    for (int i=0; i < HISTO_HEIGHT*HISTO_WIDTH; i++){
        if (gold_bins[i] != kernel_bins[i]){
            passed = 0;
            break;
        }
    }
    (passed) ? printf("\n    Test PASSED\n") : printf("\n    Test FAILED\n");

    free(gold_bins);
    free(kernel_bins);
}
