#include <unittest/unittest.h>
#include <thrust/uninitialized_fill.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename T>
__global__
void uninitialized_fill_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T val)
{
  thrust::uninitialized_fill(exec, first, last, val);
}


template<typename ExecutionPolicy>
void TestUninitializedFillDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector v(5);
  v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;
  
  T exemplar(7);
  
  uninitialized_fill_kernel<<<1,1>>>(exec, v.begin() + 1, v.begin() + 4, exemplar);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], 4);
  
  exemplar = 8;
  
  uninitialized_fill_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 3, exemplar);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], 7);
  ASSERT_EQUAL(v[4], 4);
  
  exemplar = 9;
  
  uninitialized_fill_kernel<<<1,1>>>(exec, v.begin() + 2, v.end(), exemplar);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(v[0], 8);
  ASSERT_EQUAL(v[1], 8);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], 9);
  
  exemplar = 1;
  
  uninitialized_fill_kernel<<<1,1>>>(exec, v.begin(), v.end(), exemplar);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], exemplar);
}


void TestUninitializedFillDeviceSeq()
{
  TestUninitializedFillDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUninitializedFillDeviceSeq);


void TestUninitializedFillDeviceDevice()
{
  TestUninitializedFillDevice(thrust::device);
}
DECLARE_UNITTEST(TestUninitializedFillDeviceDevice);


void TestUninitializedFillCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector v(5);
  v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;
  
  T exemplar(7);

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::uninitialized_fill(thrust::cuda::par.on(s), v.begin(), v.end(), exemplar);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], exemplar);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUninitializedFillCudaStreams);


template<typename ExecutionPolicy, typename Iterator1, typename Size, typename T, typename Iterator2>
__global__
void uninitialized_fill_n_kernel(ExecutionPolicy exec, Iterator1 first, Size n, T val, Iterator2 result)
{
  *result = thrust::uninitialized_fill_n(exec, first, n, val);
}


template<typename ExecutionPolicy>
void TestUninitializedFillNDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector v(5);
  v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;
  
  T exemplar(7);

  thrust::device_vector<Vector::iterator> iter_vec(1);
  
  uninitialized_fill_n_kernel<<<1,1>>>(exec, v.begin() + 1, 3, exemplar, iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  Vector::iterator iter = iter_vec[0];
  
  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], 4);
  ASSERT_EQUAL_QUIET(v.begin() + 4, iter);
  
  exemplar = 8;
  
  uninitialized_fill_n_kernel<<<1,1>>>(exec, v.begin() + 0, 3, exemplar, iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], 7);
  ASSERT_EQUAL(v[4], 4);
  ASSERT_EQUAL_QUIET(v.begin() + 3, iter);
  
  exemplar = 9;
  
  uninitialized_fill_n_kernel<<<1,1>>>(exec, v.begin() + 2, 3, exemplar, iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  
  ASSERT_EQUAL(v[0], 8);
  ASSERT_EQUAL(v[1], 8);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], 9);
  ASSERT_EQUAL_QUIET(v.end(), iter);
  
  exemplar = 1;
  
  uninitialized_fill_n_kernel<<<1,1>>>(exec, v.begin(), v.size(), exemplar, iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], exemplar);
  ASSERT_EQUAL_QUIET(v.end(), iter);
}


void TestUninitializedFillNDeviceSeq()
{
  TestUninitializedFillNDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUninitializedFillNDeviceSeq);


void TestUninitializedFillNDeviceDevice()
{
  TestUninitializedFillNDevice(thrust::device);
}
DECLARE_UNITTEST(TestUninitializedFillNDeviceDevice);


void TestUninitializedFillNCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector v(5);
  v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;
  
  T exemplar(7);

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::uninitialized_fill_n(thrust::cuda::par.on(s), v.begin(), v.size(), exemplar);
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(v[0], exemplar);
  ASSERT_EQUAL(v[1], exemplar);
  ASSERT_EQUAL(v[2], exemplar);
  ASSERT_EQUAL(v[3], exemplar);
  ASSERT_EQUAL(v[4], exemplar);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestUninitializedFillNCudaStreams);

