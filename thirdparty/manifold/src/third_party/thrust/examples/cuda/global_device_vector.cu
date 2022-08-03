#include <thrust/detail/config.h>
#include <thrust/device_vector.h>

// If you create a global `thrust::device_vector` with the default allocator,
// you'll get an error during program termination when the memory of the vector
// is freed, as the CUDA runtime cannot be used during program termination.
//
// To get around this, you can create your own allocator which ignores
// deallocation failures that occur because the CUDA runtime is shut down.

extern "C" cudaError_t cudaFreeIgnoreShutdown(void* ptr) {
  cudaError_t const err = cudaFree(ptr);
  if (cudaSuccess == err || cudaErrorCudartUnloading == err)
    return cudaSuccess;
  return err; 
}

typedef thrust::system::cuda::detail::cuda_memory_resource<
  cudaMalloc, 
  cudaFreeIgnoreShutdown,
  thrust::cuda::pointer<void>
> device_ignore_shutdown_memory_resource;

#if THRUST_CPP_DIALECT >= 2011
  template <typename T>
  using device_ignore_shutdown_allocator = 
    thrust::mr::stateless_resource_allocator<
      T,
      thrust::device_ptr_memory_resource<device_ignore_shutdown_memory_resource>
    >;
    
  thrust::device_vector<double, device_ignore_shutdown_allocator<double>> d;
#else
  thrust::device_vector<
    double, 
    thrust::mr::stateless_resource_allocator<
      double,
      thrust::device_ptr_memory_resource<device_ignore_shutdown_memory_resource>
    > 
  > d;
#endif

int main() {
  d.resize(25);
}

