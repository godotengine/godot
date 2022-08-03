#include <unittest/unittest.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5, typename Iterator6>
__global__
void set_intersection_by_key_kernel(ExecutionPolicy exec,
                                    Iterator1 keys_first1, Iterator1 keys_last1,
                                    Iterator2 keys_first2, Iterator2 keys_last2,
                                    Iterator3 values_first1,
                                    Iterator4 keys_result,
                                    Iterator5 values_result,
                                    Iterator6 result)
{
  *result = thrust::set_intersection_by_key(exec, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, keys_result, values_result);
}


template<typename ExecutionPolicy>
void TestSetIntersectionByKeyDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::iterator Iterator;

  Vector a_key(3), b_key(4);
  Vector a_val(3);

  a_key[0] = 0; a_key[1] = 2; a_key[2] = 4;
  a_val[0] = 0; a_val[1] = 0; a_val[2] = 0;

  b_key[0] = 0; b_key[1] = 3; b_key[2] = 3; b_key[3] = 4;

  Vector ref_key(2), ref_val(2);
  ref_key[0] = 0; ref_key[1] = 4;
  ref_val[0] = 0; ref_val[1] = 0;

  Vector result_key(2), result_val(2);

  typedef thrust::pair<Iterator,Iterator> iter_pair;
  thrust::device_vector<iter_pair> end_vec(1);

  set_intersection_by_key_kernel<<<1,1>>>(exec,
                                          a_key.begin(), a_key.end(),
                                          b_key.begin(), b_key.end(),
                                          a_val.begin(),
                                          result_key.begin(),
                                          result_val.begin(),
                                          end_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  thrust::pair<Iterator,Iterator> end = end_vec.front();

  ASSERT_EQUAL_QUIET(result_key.end(), end.first);
  ASSERT_EQUAL_QUIET(result_val.end(), end.second);
  ASSERT_EQUAL(ref_key, result_key);
  ASSERT_EQUAL(ref_val, result_val);
}


void TestSetIntersectionByKeyDeviceSeq()
{
  TestSetIntersectionByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSetIntersectionByKeyDeviceSeq);


void TestSetIntersectionByKeyDeviceDevice()
{
  TestSetIntersectionByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestSetIntersectionByKeyDeviceDevice);


void TestSetIntersectionByKeyDeviceNoSync()
{
  TestSetIntersectionByKeyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestSetIntersectionByKeyDeviceNoSync);


template<typename ExecutionPolicy>
void TestSetIntersectionByKeyCudaStreams(ExecutionPolicy policy)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::iterator Iterator;

  Vector a_key(3), b_key(4);
  Vector a_val(3);

  a_key[0] = 0; a_key[1] = 2; a_key[2] = 4;
  a_val[0] = 0; a_val[1] = 0; a_val[2] = 0;

  b_key[0] = 0; b_key[1] = 3; b_key[2] = 3; b_key[3] = 4;

  Vector ref_key(2), ref_val(2);
  ref_key[0] = 0; ref_key[1] = 4;
  ref_val[0] = 0; ref_val[1] = 0;

  Vector result_key(2), result_val(2);

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  thrust::pair<Iterator,Iterator> end =
    thrust::set_intersection_by_key(streampolicy,
                                    a_key.begin(), a_key.end(),
                                    b_key.begin(), b_key.end(),
                                    a_val.begin(),
                                    result_key.begin(),
                                    result_val.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(result_key.end(), end.first);
  ASSERT_EQUAL_QUIET(result_val.end(), end.second);
  ASSERT_EQUAL(ref_key, result_key);
  ASSERT_EQUAL(ref_val, result_val);

  cudaStreamDestroy(s);
}

void TestSetIntersectionByKeyCudaStreamsSync()
{
  TestSetIntersectionByKeyCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestSetIntersectionByKeyCudaStreamsSync);


void TestSetIntersectionByKeyCudaStreamsNoSync()
{
  TestSetIntersectionByKeyCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestSetIntersectionByKeyCudaStreamsNoSync);

