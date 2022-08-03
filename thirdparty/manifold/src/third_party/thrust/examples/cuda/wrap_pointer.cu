#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cuda.h>

int main(void)
{
    size_t N = 10;

    // obtain raw pointer to device memory
    int * raw_ptr;
    cudaMalloc((void **) &raw_ptr, N * sizeof(int));

    // wrap raw pointer with a device_ptr 
    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(raw_ptr);

    // use device_ptr in Thrust algorithms
    thrust::fill(dev_ptr, dev_ptr + N, (int) 0);

    // access device memory transparently through device_ptr
    dev_ptr[0] = 1;

    // free memory
    cudaFree(raw_ptr);

    return 0;
}
