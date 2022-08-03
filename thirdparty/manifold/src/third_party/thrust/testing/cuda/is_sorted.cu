#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename Iterator2>
__global__
void is_sorted_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Iterator2 result)
{
  *result = thrust::is_sorted(exec, first, last);
}


template<typename ExecutionPolicy>
void TestIsSortedDevice(ExecutionPolicy exec)
{
  size_t n = 1000;

  thrust::device_vector<int> v = unittest::random_integers<int>(n);

  thrust::device_vector<bool> result(1);

  v[0] = 1;
  v[1] = 0;

  is_sorted_kernel<<<1,1>>>(exec, v.begin(), v.end(), result.begin());

  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);

  thrust::sort(v.begin(), v.end());

  is_sorted_kernel<<<1,1>>>(exec, v.begin(), v.end(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);
}

void TestIsSortedDeviceSeq()
{
  TestIsSortedDevice(thrust::seq);
}
DECLARE_UNITTEST(TestIsSortedDeviceSeq);


void TestIsSortedDeviceDevice()
{
  TestIsSortedDevice(thrust::device);
}
DECLARE_UNITTEST(TestIsSortedDeviceDevice);


void TestIsSortedCudaStreams()
{
  thrust::device_vector<int> v(4);
  v[0] = 0; v[1] = 5; v[2] = 8; v[3] = 0;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par.on(s), v.begin(), v.begin() + 0), true);
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par.on(s), v.begin(), v.begin() + 1), true);
  
  // the following line crashes gcc 4.3
#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
  // do nothing
#else
  // compile this line on other compilers
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par.on(s), v.begin(), v.begin() + 2), true);
#endif // GCC

  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par.on(s), v.begin(), v.begin() + 3), true);
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par.on(s), v.begin(), v.begin() + 4), false);
  
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par.on(s), v.begin(), v.begin() + 3, thrust::less<int>()),    true);
  
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par.on(s), v.begin(), v.begin() + 1, thrust::greater<int>()), true);
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par.on(s), v.begin(), v.begin() + 4, thrust::greater<int>()), false);
  
  ASSERT_EQUAL(thrust::is_sorted(thrust::cuda::par.on(s), v.begin(), v.end()), false);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestIsSortedCudaStreams);

