#include <thrust/device_vector.h>
#include <thrust/execution_policy.h> // For thrust::device
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <cuda_runtime.h>

#include <iostream>

// This example shows how to execute a Thrust device algorithm on an explicit
// CUDA stream. The simple program below fills a vector with the numbers
// [0, 1000) (thrust::sequence) and then performs a scan operation
// (thrust::inclusive_scan) on them. Both algorithms are executed on the same
// custom CUDA stream using the CUDA execution policies.
//
// Thrust provides two execution policies that accept CUDA streams that differ
// in when/if they synchronize the stream:
// 1. thrust::cuda::par.on(stream)
//      - `stream` will *always* be synchronized before an algorithm returns.
//      - This is the default `thrust::device` policy when compiling with the
//        CUDA device backend.
// 2. thrust::cuda::par_nosync.on(stream)
//      - `stream` will only be synchronized when necessary for correctness
//        (e.g., returning a result from `thrust::reduce`). This is a hint that
//        may be ignored by an algorithm's implementation.

int main()
{
  thrust::device_vector<int> d_vec(1000);

  // Create the stream:
  cudaStream_t custom_stream;
  cudaError_t err = cudaStreamCreate(&custom_stream);
  if (err != cudaSuccess)
  {
    std::cerr << "Error creating stream: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  // Construct a new `nosync` execution policy with the custom stream
  auto nosync_exec_policy = thrust::cuda::par_nosync.on(custom_stream);

  // Fill the vector with sequential data.
  // This will execute using the custom stream and the stream will *not* be
  // synchronized before the function returns, meaning asynchronous work may
  // still be executing after returning and the contents of `d_vec` are
  // undefined. Synchronization is not needed here because the following
  // `inclusive_scan` is executed on the same stream and is therefore guaranteed
  // to be ordered after the `sequence`
  thrust::sequence(nosync_exec_policy, d_vec.begin(), d_vec.end());

  // Construct a new *synchronous* execution policy with the same custom stream
  auto sync_exec_policy = thrust::cuda::par.on(custom_stream);

  // Compute in-place inclusive sum scan of data in the vector.
  // This also executes in the custom stream, but the execution policy ensures
  // the stream is synchronized before the algorithm returns. This guarantees
  // there is no pending asynchronous work and the contents of `d_vec` are
  // immediately accessible.
  thrust::inclusive_scan(sync_exec_policy,
                         d_vec.cbegin(),
                         d_vec.cend(),
                         d_vec.begin());

  // This access is only valid because the stream has been synchronized
  int sum = d_vec.back();

  // Free the stream:
  err = cudaStreamDestroy(custom_stream);
  if (err != cudaSuccess)
  {
    std::cerr << "Error destroying stream: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  // Print the sum:
  std::cout << "sum is " << sum << std::endl;

  return 0;
}
