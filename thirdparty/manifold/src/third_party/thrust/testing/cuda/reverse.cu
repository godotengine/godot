#include <unittest/unittest.h>
#include <thrust/reverse.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator>
__global__
void reverse_kernel(ExecutionPolicy exec, Iterator first, Iterator last)
{
  thrust::reverse(exec, first, last);
}


template<typename ExecutionPolicy>
void TestReverseDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int> h_data = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  thrust::reverse(h_data.begin(), h_data.end());

  reverse_kernel<<<1,1>>>(exec, raw_pointer_cast(d_data.data()), raw_pointer_cast(d_data.data() + d_data.size()));
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_EQUAL(h_data, d_data);
};


void TestReverseDeviceSeq()
{
  TestReverseDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReverseDeviceSeq);


void TestReverseDeviceDevice()
{
  TestReverseDevice(thrust::device);
}
DECLARE_UNITTEST(TestReverseDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__
void reverse_copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  thrust::reverse_copy(exec, first, last, result);
}


template<typename ExecutionPolicy>
void TestReverseCopyDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int> h_data = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_data = h_data;

  thrust::host_vector<int> h_result(n);
  thrust::device_vector<int> d_result(n);

  thrust::reverse_copy(h_data.begin(), h_data.end(), h_result.begin());

  reverse_copy_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(h_result, d_result);
};


void TestReverseCopyDeviceSeq()
{
  TestReverseCopyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReverseCopyDeviceSeq);


void TestReverseCopyDeviceDevice()
{
  TestReverseCopyDevice(thrust::device);
}
DECLARE_UNITTEST(TestReverseCopyDeviceDevice);


void TestReverseCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  Vector data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  data[4] = 5;

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::reverse(thrust::cuda::par.on(s), data.begin(), data.end());

  cudaStreamSynchronize(s);

  Vector ref(5);
  ref[0] = 5;
  ref[1] = 4;
  ref[2] = 3;
  ref[3] = 2;
  ref[4] = 1;

  ASSERT_EQUAL(ref, data);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestReverseCudaStreams);


void TestReverseCopyCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  Vector data(5);
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  data[4] = 5;

  Vector result(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::reverse_copy(thrust::cuda::par.on(s), data.begin(), data.end(), result.begin());

  cudaStreamSynchronize(s);

  Vector ref(5);
  ref[0] = 5;
  ref[1] = 4;
  ref[2] = 3;
  ref[3] = 2;
  ref[4] = 1;

  ASSERT_EQUAL(ref, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestReverseCopyCudaStreams);

