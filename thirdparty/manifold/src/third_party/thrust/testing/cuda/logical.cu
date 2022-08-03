#include <unittest/unittest.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename Function, typename Iterator2>
__global__
void all_of_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Function f, Iterator2 result)
{
  *result = thrust::all_of(exec, first, last, f);
}


template<typename ExecutionPolicy>
void TestAllOfDevice(ExecutionPolicy exec)
{
  typedef int T;
  thrust::device_vector<T> v(3, 1);
  thrust::device_vector<bool> result(1);
  
  all_of_kernel<<<1,1>>>(exec, v.begin(), v.end(), thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);
  
  v[1] = 0;
  
  all_of_kernel<<<1,1>>>(exec, v.begin(), v.end(), thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);
  
  all_of_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 0, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);

  all_of_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 1, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);

  all_of_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 2, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);

  all_of_kernel<<<1,1>>>(exec, v.begin() + 1, v.begin() + 2, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);
}


void TestAllOfDeviceSeq()
{
  TestAllOfDevice(thrust::seq);
}
DECLARE_UNITTEST(TestAllOfDeviceSeq);


void TestAllOfDeviceDevice()
{
  TestAllOfDevice(thrust::device);
}
DECLARE_UNITTEST(TestAllOfDeviceDevice);


void TestAllOfCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector v(3, 1);

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par.on(s), v.begin(), v.end(), thrust::identity<T>()), true);
  
  v[1] = 0;
  
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par.on(s), v.begin(), v.end(), thrust::identity<T>()), false);
  
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 0, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 1, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 2, thrust::identity<T>()), false);
  ASSERT_EQUAL(thrust::all_of(thrust::cuda::par.on(s), v.begin() + 1, v.begin() + 2, thrust::identity<T>()), false);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestAllOfCudaStreams);


template<typename ExecutionPolicy, typename Iterator, typename Function, typename Iterator2>
__global__
void any_of_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Function f, Iterator2 result)
{
  *result = thrust::any_of(exec, first, last, f);
}


template<typename ExecutionPolicy>
void TestAnyOfDevice(ExecutionPolicy exec)
{
  typedef int T;
  
  thrust::device_vector<T> v(3, 1);
  thrust::device_vector<bool> result(1);
  
  any_of_kernel<<<1,1>>>(exec, v.begin(), v.end(), thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);
  
  v[1] = 0;
  
  any_of_kernel<<<1,1>>>(exec, v.begin(), v.end(), thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);
  
  any_of_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 0, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);

  any_of_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 1, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);

  any_of_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 2, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);

  any_of_kernel<<<1,1>>>(exec, v.begin() + 1, v.begin() + 2, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);
}


void TestAnyOfDeviceSeq()
{
  TestAnyOfDevice(thrust::seq);
}
DECLARE_UNITTEST(TestAnyOfDeviceSeq);


void TestAnyOfDeviceDevice()
{
  TestAnyOfDevice(thrust::device);
}
DECLARE_UNITTEST(TestAnyOfDeviceDevice);


void TestAnyOfCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector v(3, 1);

  cudaStream_t s;
  cudaStreamCreate(&s);

  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par.on(s), v.begin(), v.end(), thrust::identity<T>()), true);

  v[1] = 0;
  
  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par.on(s), v.begin(), v.end(), thrust::identity<T>()), true);

  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 0, thrust::identity<T>()), false);
  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 1, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 2, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::any_of(thrust::cuda::par.on(s), v.begin() + 1, v.begin() + 2, thrust::identity<T>()), false);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestAnyOfCudaStreams);


template<typename ExecutionPolicy, typename Iterator, typename Function, typename Iterator2>
__global__
void none_of_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Function f, Iterator2 result)
{
  *result = thrust::none_of(exec, first, last, f);
}


template<typename ExecutionPolicy>
void TestNoneOfDevice(ExecutionPolicy exec)
{
  typedef int T;
  
  thrust::device_vector<T> v(3, 1);
  thrust::device_vector<bool> result(1);
  
  none_of_kernel<<<1,1>>>(exec, v.begin(), v.end(), thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);
  
  v[1] = 0;
  
  none_of_kernel<<<1,1>>>(exec, v.begin(), v.end(), thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);
  
  none_of_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 0, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);

  none_of_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 1, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);

  none_of_kernel<<<1,1>>>(exec, v.begin() + 0, v.begin() + 2, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(false, result[0]);

  none_of_kernel<<<1,1>>>(exec, v.begin() + 1, v.begin() + 2, thrust::identity<T>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(true, result[0]);
}


void TestNoneOfDeviceSeq()
{
  TestNoneOfDevice(thrust::seq);
}
DECLARE_UNITTEST(TestNoneOfDeviceSeq);


void TestNoneOfDeviceDevice()
{
  TestNoneOfDevice(thrust::device);
}
DECLARE_UNITTEST(TestNoneOfDeviceDevice);


void TestNoneOfCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector v(3, 1);

  cudaStream_t s;
  cudaStreamCreate(&s);

  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par.on(s), v.begin(), v.end(), thrust::identity<T>()), false);

  v[1] = 0;
  
  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par.on(s), v.begin(), v.end(), thrust::identity<T>()), false);

  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 0, thrust::identity<T>()), true);
  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 1, thrust::identity<T>()), false);
  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 2, thrust::identity<T>()), false);
  ASSERT_EQUAL(thrust::none_of(thrust::cuda::par.on(s), v.begin() + 1, v.begin() + 2, thrust::identity<T>()), true);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestNoneOfCudaStreams);

