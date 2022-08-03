#include <unittest/unittest.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename T, typename Iterator2>
__global__
void count_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T value, Iterator2 result)
{
  *result = thrust::count(exec, first, last, value);
}


template<typename T, typename ExecutionPolicy>
void TestCountDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::device_vector<size_t> d_result(1);
  
  size_t h_result = thrust::count(h_data.begin(), h_data.end(), T(5));

  count_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), T(5), d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_EQUAL(h_result, d_result[0]);
}


template<typename T>
void TestCountDeviceSeq(const size_t n)
{
  TestCountDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestCountDeviceSeq);


template<typename T>
void TestCountDeviceDevice(const size_t n)
{
  TestCountDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestCountDeviceDevice);


template<typename ExecutionPolicy, typename Iterator, typename Predicate, typename Iterator2>
__global__
void count_if_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::count_if(exec, first, last, pred);
}


template<typename T>
struct greater_than_five
{
  __host__ __device__ bool operator()(const T &x) const {return x > 5;}
};


template<typename T, typename ExecutionPolicy>
void TestCountIfDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::device_vector<size_t> d_result(1);
  
  size_t h_result = thrust::count_if(h_data.begin(), h_data.end(), greater_than_five<T>());
  count_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), greater_than_five<T>(), d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_EQUAL(h_result, d_result[0]);
}


template<typename T>
void TestCountIfDeviceSeq(const size_t n)
{
  TestCountIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestCountIfDeviceSeq);


template<typename T>
void TestCountIfDeviceDevice(const size_t n)
{
  TestCountIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestCountIfDeviceDevice);


void TestCountCudaStreams()
{
  thrust::device_vector<int> data(5);
  data[0] = 1; data[1] = 1; data[2] = 0; data[3] = 0; data[4] = 1;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  ASSERT_EQUAL(thrust::count(thrust::cuda::par.on(s), data.begin(), data.end(), 0), 2);
  ASSERT_EQUAL(thrust::count(thrust::cuda::par.on(s), data.begin(), data.end(), 1), 3);
  ASSERT_EQUAL(thrust::count(thrust::cuda::par.on(s), data.begin(), data.end(), 2), 0);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestCountCudaStreams);

