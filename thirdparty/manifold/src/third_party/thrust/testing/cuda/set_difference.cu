#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
__global__
void set_difference_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 result1, Iterator4 result2)
{
  *result2 = thrust::set_difference(exec, first1, last1, first2, last2, result1);
}


template<typename ExecutionPolicy>
void TestSetDifferenceDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::iterator Iterator;

  Vector a(4), b(5);

  a[0] = 0; a[1] = 2; a[2] = 4; a[3] = 5;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4; b[4] = 6;

  Vector ref(2);
  ref[0] = 2; ref[1] = 5;

  Vector result(2);

  thrust::device_vector<Iterator> end_vec(1);

  set_difference_kernel<<<1,1>>>(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin(), end_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  Iterator end = end_vec.front();

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}


void TestSetDifferenceDeviceSeq()
{
  TestSetDifferenceDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSetDifferenceDeviceSeq);


void TestSetDifferenceDeviceDevice()
{
  TestSetDifferenceDevice(thrust::device);
}
DECLARE_UNITTEST(TestSetDifferenceDeviceDevice);


void TestSetDifferenceCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::iterator Iterator;

  Vector a(4), b(5);

  a[0] = 0; a[1] = 2; a[2] = 4; a[3] = 5;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4; b[4] = 6;

  Vector ref(2);
  ref[0] = 2; ref[1] = 5;

  Vector result(2);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Iterator end = thrust::set_difference(thrust::cuda::par.on(s), a.begin(), a.end(), b.begin(), b.end(), result.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSetDifferenceCudaStreams);

