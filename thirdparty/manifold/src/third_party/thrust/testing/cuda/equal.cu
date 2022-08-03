#include <unittest/unittest.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void equal_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 result)
{
  *result = thrust::equal(exec, first1, last1, first2);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename BinaryPredicate, typename Iterator3>
__global__
void equal_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, BinaryPredicate pred, Iterator3 result)
{
  *result = thrust::equal(exec, first1, last1, first2, pred);
}


template<typename T, typename ExecutionPolicy>
void TestEqualDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::device_vector<T> d_data1 = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data2 = unittest::random_samples<T>(n);
  thrust::device_vector<bool> d_result(1, false);
  
  //empty ranges
  equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.begin(), d_data1.begin(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(d_result[0], true);
  
  //symmetric cases
  equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.end(), d_data1.begin(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(d_result[0], true);
  
  if(n > 0)
  {
    d_data1[0] = 0; d_data2[0] = 1;
    
    //different vectors
    equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.end(), d_data2.begin(), d_result.begin());
    {
      cudaError_t const err = cudaDeviceSynchronize();
      ASSERT_EQUAL(cudaSuccess, err);
    }

    ASSERT_EQUAL(d_result[0], false);
    
    //different predicates
    equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::less<T>(), d_result.begin());
    {
      cudaError_t const err = cudaDeviceSynchronize();
      ASSERT_EQUAL(cudaSuccess, err);
    }

    ASSERT_EQUAL(d_result[0], true);

    equal_kernel<<<1,1>>>(exec, d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::greater<T>(), d_result.begin());
    {
      cudaError_t const err = cudaDeviceSynchronize();
      ASSERT_EQUAL(cudaSuccess, err);
    }

    ASSERT_EQUAL(d_result[0], false);
  }
}


template<typename T>
void TestEqualDeviceSeq(const size_t n)
{
  TestEqualDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestEqualDeviceSeq);


template<typename T>
void TestEqualDeviceDevice(const size_t n)
{
  TestEqualDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestEqualDeviceDevice);


void TestEqualCudaStreams()
{
  thrust::device_vector<int> v1(5);
  thrust::device_vector<int> v2(5);
  v1[0] = 5; v1[1] = 2; v1[2] = 0; v1[3] = 0; v1[4] = 0;
  v2[0] = 5; v2[1] = 2; v2[2] = 0; v2[3] = 6; v2[4] = 1;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  ASSERT_EQUAL(thrust::equal(thrust::cuda::par.on(s), v1.begin(), v1.end(), v1.begin()), true);
  ASSERT_EQUAL(thrust::equal(thrust::cuda::par.on(s), v1.begin(), v1.end(), v2.begin()), false);
  ASSERT_EQUAL(thrust::equal(thrust::cuda::par.on(s), v2.begin(), v2.end(), v2.begin()), true);
  
  ASSERT_EQUAL(thrust::equal(thrust::cuda::par.on(s), v1.begin(), v1.begin() + 0, v1.begin()), true);
  ASSERT_EQUAL(thrust::equal(thrust::cuda::par.on(s), v1.begin(), v1.begin() + 1, v1.begin()), true);
  ASSERT_EQUAL(thrust::equal(thrust::cuda::par.on(s), v1.begin(), v1.begin() + 3, v2.begin()), true);
  ASSERT_EQUAL(thrust::equal(thrust::cuda::par.on(s), v1.begin(), v1.begin() + 4, v2.begin()), false);
  
  ASSERT_EQUAL(thrust::equal(thrust::cuda::par.on(s), v1.begin(), v1.end(), v2.begin(), thrust::less_equal<int>()), true);
  ASSERT_EQUAL(thrust::equal(thrust::cuda::par.on(s), v1.begin(), v1.end(), v2.begin(), thrust::greater<int>()),    false);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestEqualCudaStreams);

