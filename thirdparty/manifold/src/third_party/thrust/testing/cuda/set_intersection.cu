#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
__global__
void set_intersection_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1,
                             Iterator2 first2, Iterator2 last2,
                             Iterator3 result1,
                             Iterator4 result2)
{
  *result2 = thrust::set_intersection(exec, first1, last1, first2, last2, result1);
}


template<typename ExecutionPolicy>
void TestSetIntersectionDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::iterator Iterator;

  Vector a(3), b(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;

  Vector ref(2);
  ref[0] = 0; ref[1] = 4;

  Vector result(2);
  thrust::device_vector<Iterator> end_vec(1);

  set_intersection_kernel<<<1,1>>>(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin(), end_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  Iterator end = end_vec.front();

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}


void TestSetIntersectionDeviceSeq()
{
  TestSetIntersectionDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSetIntersectionDeviceSeq);


void TestSetIntersectionDeviceDevice()
{
  TestSetIntersectionDevice(thrust::device);
}
DECLARE_UNITTEST(TestSetIntersectionDeviceDevice);


void TestSetIntersectionDeviceNoSync()
{
  TestSetIntersectionDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestSetIntersectionDeviceNoSync);


template<typename ExecutionPolicy>
void TestSetIntersectionCudaStreams(ExecutionPolicy policy)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::iterator Iterator;

  Vector a(3), b(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;

  Vector ref(2);
  ref[0] = 0; ref[1] = 4;

  Vector result(2);

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  Iterator end = thrust::set_intersection(streampolicy,
                                          a.begin(), a.end(),
                                          b.begin(), b.end(),
                                          result.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);

  cudaStreamDestroy(s);
}

void TestSetIntersectionCudaStreamsSync()
{
  TestSetIntersectionCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestSetIntersectionCudaStreamsSync);


void TestSetIntersectionCudaStreamsNoSync()
{
  TestSetIntersectionCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestSetIntersectionCudaStreamsNoSync);

