#include <unittest/unittest.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__
void copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  thrust::copy(exec, first, last, result);
}


template<typename T, typename ExecutionPolicy>
void TestCopyDevice(ExecutionPolicy exec, size_t n)
{
  thrust::host_vector<T>   h_src = unittest::random_integers<T>(n);
  thrust::host_vector<T>   h_dst(n);

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n);
  
  thrust::copy(h_src.begin(), h_src.end(), h_dst.begin());
  copy_kernel<<<1,1>>>(exec, d_src.begin(), d_src.end(), d_dst.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_dst, d_dst);
}


template<typename T>
void TestCopyDeviceSeq(size_t n)
{
  TestCopyDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestCopyDeviceSeq);


template<typename T>
void TestCopyDeviceDevice(size_t n)
{
  TestCopyDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestCopyDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Size, typename Iterator2>
__global__
void copy_n_kernel(ExecutionPolicy exec, Iterator1 first, Size n, Iterator2 result)
{
  thrust::copy_n(exec, first, n, result);
}


template<typename T, typename ExecutionPolicy>
void TestCopyNDevice(ExecutionPolicy exec, size_t n)
{
  thrust::host_vector<T>   h_src = unittest::random_integers<T>(n);
  thrust::host_vector<T>   h_dst(n);

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n);
  
  thrust::copy_n(h_src.begin(), h_src.size(), h_dst.begin());
  copy_n_kernel<<<1,1>>>(exec, d_src.begin(), d_src.size(), d_dst.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_dst, d_dst);
}


template<typename T>
void TestCopyNDeviceSeq(size_t n)
{
  TestCopyNDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestCopyNDeviceSeq);


template<typename T>
void TestCopyNDeviceDevice(size_t n)
{
  TestCopyNDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestCopyNDeviceDevice);

