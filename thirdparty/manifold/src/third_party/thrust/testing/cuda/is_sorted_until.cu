#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__
void is_sorted_until_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::is_sorted_until(exec, first, last);
}


template<typename ExecutionPolicy>
void TestIsSortedUntilDevice(ExecutionPolicy exec)
{
  size_t n = 1000;

  thrust::device_vector<int> v = unittest::random_integers<int>(n);

  typedef typename thrust::device_vector<int>::iterator iter_type;

  thrust::device_vector<iter_type> result(1);

  v[0] = 1;
  v[1] = 0;
  
  is_sorted_until_kernel<<<1,1>>>(exec, v.begin(), v.end(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL_QUIET(v.begin() + 1, (iter_type)result[0]);
  
  thrust::sort(v.begin(), v.end());
  
  is_sorted_until_kernel<<<1,1>>>(exec, v.begin(), v.end(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL_QUIET(v.end(), (iter_type)result[0]);
}


void TestIsSortedUntilDeviceSeq()
{
  TestIsSortedUntilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestIsSortedUntilDeviceSeq);


void TestIsSortedUntilDeviceDevice()
{
  TestIsSortedUntilDevice(thrust::device);
}
DECLARE_UNITTEST(TestIsSortedUntilDeviceDevice);


void TestIsSortedUntilCudaStreams()
{
  typedef thrust::device_vector<int> Vector;

  typedef Vector::value_type T;
  typedef Vector::iterator Iterator;

  cudaStream_t s;
  cudaStreamCreate(&s);

  Vector v(4);
  v[0] = 0; v[1] = 5; v[2] = 8; v[3] = 0;

  Iterator first = v.begin();

  Iterator last  = v.begin() + 0;
  Iterator ref = last;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last));

  last = v.begin() + 1;
  ref = last;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last));

  last = v.begin() + 2;
  ref = last;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last));

  last = v.begin() + 3;
  ref = v.begin() + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last));

  last = v.begin() + 4;
  ref = v.begin() + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last));

  last = v.begin() + 3;
  ref = v.begin() + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last, thrust::less<T>()));

  last = v.begin() + 4;
  ref = v.begin() + 3;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last, thrust::less<T>()));

  last = v.begin() + 1;
  ref = v.begin() + 1;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last, thrust::greater<T>()));

  last = v.begin() + 4;
  ref = v.begin() + 1;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last, thrust::greater<T>()));

  first = v.begin() + 2;
  last = v.begin() + 4;
  ref = v.begin() + 4;
  ASSERT_EQUAL_QUIET(ref, thrust::is_sorted_until(thrust::cuda::par.on(s), first, last, thrust::greater<T>()));

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestIsSortedUntilCudaStreams);

