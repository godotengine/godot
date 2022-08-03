#include <unittest/unittest.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename Iterator2>
__global__
void min_element_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Iterator2 result)
{
  *result = thrust::min_element(exec, first, last);
}


template<typename ExecutionPolicy, typename Iterator, typename BinaryPredicate, typename Iterator2>
__global__
void min_element_kernel(ExecutionPolicy exec, Iterator first, Iterator last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::min_element(exec, first, last, pred);
}


template<typename ExecutionPolicy>
void TestMinElementDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int> h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  typedef typename thrust::device_vector<int>::iterator iter_type;

  thrust::device_vector<iter_type> d_result(1);
  
  typename thrust::host_vector<int>::iterator   h_min = thrust::min_element(h_data.begin(), h_data.end());

  min_element_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_min - h_data.begin(), (iter_type)d_result[0] - d_data.begin());

  typename thrust::host_vector<int>::iterator   h_max = thrust::min_element(h_data.begin(), h_data.end(), thrust::greater<int>());

  min_element_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), thrust::greater<int>(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_max - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
}


void TestMinElementDeviceSeq()
{
  TestMinElementDevice(thrust::seq);
}
DECLARE_UNITTEST(TestMinElementDeviceSeq);


void TestMinElementDeviceDevice()
{
  TestMinElementDevice(thrust::device);
}
DECLARE_UNITTEST(TestMinElementDeviceDevice);


void TestMinElementCudaStreams()
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

  cudaStream_t s;
  cudaStreamCreate(&s);

  ASSERT_EQUAL( *thrust::min_element(thrust::cuda::par.on(s), data.begin(), data.end()), 1);
  ASSERT_EQUAL( thrust::min_element(thrust::cuda::par.on(s), data.begin(), data.end()) - data.begin(), 2);
  
  ASSERT_EQUAL( *thrust::min_element(thrust::cuda::par.on(s), data.begin(), data.end(), thrust::greater<T>()), 5);
  ASSERT_EQUAL( thrust::min_element(thrust::cuda::par.on(s), data.begin(), data.end(), thrust::greater<T>()) - data.begin(), 1);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestMinElementCudaStreams);

void TestMinElementDevicePointer()
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
  ASSERT_EQUAL( thrust::min_element(thrust::device, raw_ptr, raw_ptr+n) - raw_ptr, 2);
  ASSERT_EQUAL( thrust::min_element(thrust::device, raw_ptr, raw_ptr+n, thrust::greater<T>()) - raw_ptr, 1);
}
DECLARE_UNITTEST(TestMinElementDevicePointer);
