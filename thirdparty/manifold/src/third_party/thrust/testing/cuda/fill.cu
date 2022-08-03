#include <unittest/unittest.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <algorithm>

template<typename ExecutionPolicy, typename Iterator, typename T>
__global__
void fill_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T value)
{
  thrust::fill(exec, first, last, value);
}


template<typename T, typename ExecutionPolicy>
void TestFillDevice(ExecutionPolicy exec, size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::fill(h_data.begin() + std::min((size_t)1, n), h_data.begin() + std::min((size_t)3, n), (T) 0);

  fill_kernel<<<1,1>>>(exec, d_data.begin() + std::min((size_t)1, n), d_data.begin() + std::min((size_t)3, n), (T) 0);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill(h_data.begin() + std::min((size_t)117, n), h_data.begin() + std::min((size_t)367, n), (T) 1);

  fill_kernel<<<1,1>>>(exec, d_data.begin() + std::min((size_t)117, n), d_data.begin() + std::min((size_t)367, n), (T) 1);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill(h_data.begin() + std::min((size_t)8, n), h_data.begin() + std::min((size_t)259, n), (T) 2);

  fill_kernel<<<1,1>>>(exec, d_data.begin() + std::min((size_t)8, n), d_data.begin() + std::min((size_t)259, n), (T) 2);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill(h_data.begin() + std::min((size_t)3, n), h_data.end(), (T) 3);

  fill_kernel<<<1,1>>>(exec, d_data.begin() + std::min((size_t)3, n), d_data.end(), (T) 3);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill(h_data.begin(), h_data.end(), (T) 4);

  fill_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), (T) 4);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_data, d_data);
}

template<typename T>
void TestFillDeviceSeq(size_t n)
{
  TestFillDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillDeviceSeq);

template<typename T>
void TestFillDeviceDevice(size_t n)
{
  TestFillDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillDeviceDevice);


template<typename ExecutionPolicy, typename Iterator, typename Size, typename T>
__global__
void fill_n_kernel(ExecutionPolicy exec, Iterator first, Size n, T value)
{
  thrust::fill_n(exec, first, n, value);
}


template<typename T, typename ExecutionPolicy>
void TestFillNDevice(ExecutionPolicy exec, size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  size_t begin_offset = std::min<size_t>(1,n);

  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t)3, n) - begin_offset, (T) 0);

  fill_n_kernel<<<1,1>>>(exec, d_data.begin() + begin_offset, std::min((size_t)3, n) - begin_offset, (T) 0);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
 
  ASSERT_EQUAL(h_data, d_data);
  
  begin_offset = std::min<size_t>(117, n);

  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t)367, n) - begin_offset, (T) 1);

  fill_n_kernel<<<1,1>>>(exec, d_data.begin() + begin_offset, std::min((size_t)367, n) - begin_offset, (T) 1);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_data, d_data);
  
  begin_offset = std::min<size_t>(8, n);

  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t)259, n) - begin_offset, (T) 2);

  fill_n_kernel<<<1,1>>>(exec, d_data.begin() + begin_offset, std::min((size_t)259, n) - begin_offset, (T) 2);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_data, d_data);
  
  begin_offset = std::min<size_t>(3, n);

  thrust::fill_n(h_data.begin() + begin_offset, h_data.size() - begin_offset, (T) 3);

  fill_n_kernel<<<1,1>>>(exec, d_data.begin() + begin_offset, d_data.size() - begin_offset, (T) 3);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_data, d_data);
  
  thrust::fill_n(h_data.begin(), h_data.size(), (T) 4);

  fill_n_kernel<<<1,1>>>(exec, d_data.begin(), d_data.size(), (T) 4);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_data, d_data);
}

template<typename T>
void TestFillNDeviceSeq(size_t n)
{
  TestFillNDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillNDeviceSeq);

template<typename T>
void TestFillNDeviceDevice(size_t n)
{
  TestFillNDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillNDeviceDevice);

void TestFillCudaStreams()
{
  thrust::device_vector<int> v(5);
  v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::fill(thrust::cuda::par.on(s), v.begin() + 1, v.begin() + 4, 7);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 7);
  ASSERT_EQUAL(v[2], 7);
  ASSERT_EQUAL(v[3], 7);
  ASSERT_EQUAL(v[4], 4);
  
  thrust::fill(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 3, 8);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(v[0], 8);
  ASSERT_EQUAL(v[1], 8);
  ASSERT_EQUAL(v[2], 8);
  ASSERT_EQUAL(v[3], 7);
  ASSERT_EQUAL(v[4], 4);
  
  thrust::fill(thrust::cuda::par.on(s), v.begin() + 2, v.end(), 9);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(v[0], 8);
  ASSERT_EQUAL(v[1], 8);
  ASSERT_EQUAL(v[2], 9);
  ASSERT_EQUAL(v[3], 9);
  ASSERT_EQUAL(v[4], 9);
  
  thrust::fill(thrust::cuda::par.on(s), v.begin(), v.end(), 1);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(v[0], 1);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 1);
  ASSERT_EQUAL(v[3], 1);
  ASSERT_EQUAL(v[4], 1);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestFillCudaStreams);

