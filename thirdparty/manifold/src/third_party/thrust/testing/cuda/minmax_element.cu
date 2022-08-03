#include <unittest/unittest.h>
#include <thrust/extrema.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__
void minmax_element_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::minmax_element(exec, first, last);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename BinaryPredicate>
__global__
void minmax_element_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::minmax_element(exec, first, last, pred);
}


template<typename ExecutionPolicy>
void TestMinMaxElementDevice(ExecutionPolicy exec)
{
  size_t n = 1000;

  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  typename thrust::host_vector<int>::iterator   h_min;
  typename thrust::host_vector<int>::iterator   h_max;
  typename thrust::device_vector<int>::iterator d_min;
  typename thrust::device_vector<int>::iterator d_max;

  typedef thrust::pair<
    typename thrust::device_vector<int>::iterator,
    typename thrust::device_vector<int>::iterator
  > pair_type;

  thrust::device_vector<pair_type> d_result(1);
  
  h_min = thrust::minmax_element(h_data.begin(), h_data.end()).first;
  h_max = thrust::minmax_element(h_data.begin(), h_data.end()).second;

  d_min = thrust::minmax_element(d_data.begin(), d_data.end()).first;
  d_max = thrust::minmax_element(d_data.begin(), d_data.end()).second;

  minmax_element_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  d_min = ((pair_type)d_result[0]).first;
  d_max = ((pair_type)d_result[0]).second;
  
  ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
  ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
  
  h_max = thrust::minmax_element(h_data.begin(), h_data.end(), thrust::greater<int>()).first;
  h_min = thrust::minmax_element(h_data.begin(), h_data.end(), thrust::greater<int>()).second;

  minmax_element_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), thrust::greater<int>(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  d_max = ((pair_type)d_result[0]).first;
  d_min = ((pair_type)d_result[0]).second;
  
  ASSERT_EQUAL(h_min - h_data.begin(), d_min - d_data.begin());
  ASSERT_EQUAL(h_max - h_data.begin(), d_max - d_data.begin());
}


void TestMinMaxElementDeviceSeq()
{
  TestMinMaxElementDevice(thrust::seq);
}
DECLARE_UNITTEST(TestMinMaxElementDeviceSeq);


void TestMinMaxElementDeviceDevice()
{
  TestMinMaxElementDevice(thrust::device);
}
DECLARE_UNITTEST(TestMinMaxElementDeviceDevice);


void TestMinMaxElementCudaStreams()
{
  typedef thrust::device_vector<int> Vector;

  Vector data(6);
  data[0] = 3;
  data[1] = 5;
  data[2] = 1;
  data[3] = 2;
  data[4] = 5;
  data[5] = 1;

  cudaStream_t s;
  cudaStreamCreate(&s);

  ASSERT_EQUAL( *thrust::minmax_element(thrust::cuda::par.on(s), data.begin(), data.end()).first,  1);
  ASSERT_EQUAL( *thrust::minmax_element(thrust::cuda::par.on(s), data.begin(), data.end()).second, 5);
  ASSERT_EQUAL(  thrust::minmax_element(thrust::cuda::par.on(s), data.begin(), data.end()).first  - data.begin(), 2);
  ASSERT_EQUAL(  thrust::minmax_element(thrust::cuda::par.on(s), data.begin(), data.end()).second - data.begin(), 1);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestMinMaxElementCudaStreams);

void TestMinMaxElementDevicePointer()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(6);
  data[0] = 3;
  data[1] = 5;
  data[2] = 1;
  data[3] = 2;
  data[4] = 5;
  data[5] = 1;

  T* raw_ptr = thrust::raw_pointer_cast(data.data());
  size_t n = data.size();
  ASSERT_EQUAL( thrust::minmax_element(thrust::device, raw_ptr, raw_ptr+n).first - raw_ptr,  2);
  ASSERT_EQUAL( thrust::minmax_element(thrust::device, raw_ptr, raw_ptr+n).second - raw_ptr, 1);
}
DECLARE_UNITTEST(TestMinMaxElementDevicePointer);

