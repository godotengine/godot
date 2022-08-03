#include <unittest/unittest.h>
#include <thrust/generate.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename Function>
__global__
void generate_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Function f)
{
  thrust::generate(exec, first, last, f);
}


template<typename T>
struct return_value
{
  T val;
  
  return_value(void){}
  return_value(T v):val(v){}
  
  __host__ __device__
  T operator()(void){ return val; }
};


template<typename T, typename ExecutionPolicy>
void TestGenerateDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);
  
  T value = 13;
  return_value<T> f(value);
  
  thrust::generate(h_result.begin(), h_result.end(), f);

  generate_kernel<<<1,1>>>(exec, d_result.begin(), d_result.end(), f);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_result, d_result);
}


template<typename T>
void TestGenerateDeviceSeq(const size_t n)
{
  TestGenerateDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateDeviceSeq);


template<typename T>
void TestGenerateDeviceDevice(const size_t n)
{
  TestGenerateDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateDeviceDevice);


void TestGenerateCudaStreams()
{
  thrust::device_vector<int> result(5);
  
  int value = 13;
  
  return_value<int> f(value);

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::generate(thrust::cuda::par.on(s), result.begin(), result.end(), f);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(result[0], value);
  ASSERT_EQUAL(result[1], value);
  ASSERT_EQUAL(result[2], value);
  ASSERT_EQUAL(result[3], value);
  ASSERT_EQUAL(result[4], value);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestGenerateCudaStreams);


template<typename ExecutionPolicy, typename Iterator, typename Size, typename Function>
__global__
void generate_n_kernel(ExecutionPolicy exec, Iterator first, Size n, Function f)
{
  thrust::generate_n(exec, first, n, f);
}


template<typename T, typename ExecutionPolicy>
void TestGenerateNDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);
  
  T value = 13;
  return_value<T> f(value);
  
  thrust::generate_n(h_result.begin(), h_result.size(), f);

  generate_n_kernel<<<1,1>>>(exec, d_result.begin(), d_result.size(), f);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_result, d_result);
}


template<typename T>
void TestGenerateNDeviceSeq(const size_t n)
{
  TestGenerateNDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateNDeviceSeq);


template<typename T>
void TestGenerateNDeviceDevice(const size_t n)
{
  TestGenerateNDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateNDeviceDevice);


void TestGenerateNCudaStreams()
{
  thrust::device_vector<int> result(5);
  
  int value = 13;
  
  return_value<int> f(value);

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::generate_n(thrust::cuda::par.on(s), result.begin(), result.size(), f);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(result[0], value);
  ASSERT_EQUAL(result[1], value);
  ASSERT_EQUAL(result[2], value);
  ASSERT_EQUAL(result[3], value);
  ASSERT_EQUAL(result[4], value);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestGenerateNCudaStreams);

