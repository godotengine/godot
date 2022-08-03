#include <unittest/unittest.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Function1, typename Function2, typename Iterator3>
__global__
void transform_inclusive_scan_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Function1 f1, Function2 f2, Iterator3 result2)
{
  *result2 = thrust::transform_inclusive_scan(exec, first, last, result1, f1, f2);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Function1, typename T, typename Function2, typename Iterator3>
__global__
void transform_exclusive_scan_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, Function1 f1, T init, Function2 f2, Iterator3 result2)
{
  *result2 = thrust::transform_exclusive_scan(exec, first, last, result, f1, init, f2);
}


template<typename ExecutionPolicy>
void TestTransformScanDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input(5);
  Vector ref(5);
  Vector output(5);
  
  input[0] = 1; input[1] = 3; input[2] = -2; input[3] = 4; input[4] = -5;
  
  Vector input_copy(input);

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  // inclusive scan
  transform_inclusive_scan_kernel<<<1,1>>>(exec, input.begin(), input.end(), output.begin(), thrust::negate<T>(), thrust::plus<T>(), iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ref[0] = -1; ref[1] = -4; ref[2] = -2; ref[3] = -6; ref[4] = -1;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(ref, output);
  
  // exclusive scan with 0 init
  transform_exclusive_scan_kernel<<<1,1>>>(exec, input.begin(), input.end(), output.begin(), thrust::negate<T>(), 0, thrust::plus<T>(), iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ref[0] = 0; ref[1] = -1; ref[2] = -4; ref[3] = -2; ref[4] = -6;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(ref, output);
  
  // exclusive scan with nonzero init
  transform_exclusive_scan_kernel<<<1,1>>>(exec, input.begin(), input.end(), output.begin(), thrust::negate<T>(), 3, thrust::plus<T>(), iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ref[0] = 3; ref[1] = 2; ref[2] = -1; ref[3] = 1; ref[4] = -3;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(ref, output);
  
  // inplace inclusive scan
  input = input_copy;
  transform_inclusive_scan_kernel<<<1,1>>>(exec, input.begin(), input.end(), input.begin(), thrust::negate<T>(), thrust::plus<T>(), iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ref[0] = -1; ref[1] = -4; ref[2] = -2; ref[3] = -6; ref[4] = -1;
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(ref, input);
  
  // inplace exclusive scan with init
  input = input_copy;
  transform_exclusive_scan_kernel<<<1,1>>>(exec, input.begin(), input.end(), input.begin(), thrust::negate<T>(), 3, thrust::plus<T>(), iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ref[0] = 3; ref[1] = 2; ref[2] = -1; ref[3] = 1; ref[4] = -3;
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(ref, input);
}


void TestTransformScanDeviceSeq()
{
  TestTransformScanDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformScanDeviceSeq);


void TestTransformScanDeviceDevice()
{
  TestTransformScanDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformScanDeviceDevice);


void TestTransformScanCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector::iterator iter;

  Vector input(5);
  Vector result(5);
  Vector output(5);

  input[0] = 1; input[1] = 3; input[2] = -2; input[3] = 4; input[4] = -5;

  Vector input_copy(input);

  cudaStream_t s;
  cudaStreamCreate(&s);

  // inclusive scan
  iter = thrust::transform_inclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), thrust::negate<T>(), thrust::plus<T>());
  cudaStreamSynchronize(s);

  result[0] = -1; result[1] = -4; result[2] = -2; result[3] = -6; result[4] = -1;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(output, result);
  
  // exclusive scan with 0 init
  iter = thrust::transform_exclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), thrust::negate<T>(), 0, thrust::plus<T>());
  cudaStreamSynchronize(s);

  result[0] = 0; result[1] = -1; result[2] = -4; result[3] = -2; result[4] = -6;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(output, result);
  
  // exclusive scan with nonzero init
  iter = thrust::transform_exclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
  cudaStreamSynchronize(s);

  result[0] = 3; result[1] = 2; result[2] = -1; result[3] = 1; result[4] = -3;
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input,  input_copy);
  ASSERT_EQUAL(output, result);
  
  // inplace inclusive scan
  input = input_copy;
  iter = thrust::transform_inclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), input.begin(), thrust::negate<T>(), thrust::plus<T>());
  cudaStreamSynchronize(s);

  result[0] = -1; result[1] = -4; result[2] = -2; result[3] = -6; result[4] = -1;
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace exclusive scan with init
  input = input_copy;
  iter = thrust::transform_exclusive_scan(thrust::cuda::par.on(s), input.begin(), input.end(), input.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
  cudaStreamSynchronize(s);

  result[0] = 3; result[1] = 2; result[2] = -1; result[3] = 1; result[4] = -3;
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestTransformScanCudaStreams);

