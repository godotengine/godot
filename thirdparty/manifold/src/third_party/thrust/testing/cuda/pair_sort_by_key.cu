#include <unittest/unittest.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void stable_sort_by_key_kernel(ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 is_supported)
{
#if (__CUDA_ARCH__ >= 200)
  *is_supported = true;
  thrust::stable_sort_by_key(exec, keys_first, keys_last, values_first);
#else
  *is_supported = false;
#endif
}


struct make_pair_functor
{
  template<typename T1, typename T2>
  __host__ __device__
    thrust::pair<T1,T2> operator()(const T1 &x, const T2 &y)
  {
    return thrust::make_pair(x,y);
  } // end operator()()
}; // end make_pair_functor


template<typename ExecutionPolicy>
void TestPairStableSortByKeyDevice(ExecutionPolicy exec)
{
  size_t n = 10000;
  typedef thrust::pair<int,int> P;

  // host arrays
  thrust::host_vector<int>   h_p1 = unittest::random_integers<int>(n);
  thrust::host_vector<int>   h_p2 = unittest::random_integers<int>(n);
  thrust::host_vector<P>   h_pairs(n);

  thrust::host_vector<int> h_values(n);
  thrust::sequence(h_values.begin(), h_values.end());

  // zip up pairs on the host
  thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

  // device arrays
  thrust::device_vector<P>   d_pairs = h_pairs;
  thrust::device_vector<int> d_values = h_values;

  thrust::device_vector<bool> is_supported(1);

  // sort on the device
  stable_sort_by_key_kernel<<<1,1>>>(exec, d_pairs.begin(), d_pairs.end(), d_values.begin(), is_supported.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  if(is_supported[0])
  {
    // sort on the host
    thrust::stable_sort_by_key(h_pairs.begin(), h_pairs.end(), h_values.begin());

    ASSERT_EQUAL_QUIET(h_pairs,  d_pairs);
    ASSERT_EQUAL(h_values, d_values);
  }
};


void TestPairStableSortByKeyDeviceSeq()
{
  TestPairStableSortByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestPairStableSortByKeyDeviceSeq);


void TestPairStableSortByKeyDeviceDevice()
{
  TestPairStableSortByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestPairStableSortByKeyDeviceDevice);

