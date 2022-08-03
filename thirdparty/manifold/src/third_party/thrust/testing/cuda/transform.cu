#include <unittest/unittest.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Function, typename Iterator3>
__global__
void transform_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Function f, Iterator3 result2)
{
  *result2 = thrust::transform(exec, first, last, result1, f);
}


template<typename ExecutionPolicy>
void TestTransformUnaryDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input(3);
  Vector output(3);
  Vector result(3);
  input[0]  =  1; input[1]  = -2; input[2]  =  3;
  result[0] = -1; result[1] =  2; result[2] = -3;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  transform_kernel<<<1,1>>>(exec, input.begin(), input.end(), output.begin(), thrust::negate<T>(), iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];
  
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformUnaryDeviceSeq()
{
  TestTransformUnaryDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformUnaryDeviceSeq);

void TestTransformUnaryDeviceDevice()
{
  TestTransformUnaryDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformUnaryDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Function, typename Predicate, typename Iterator3>
__global__
void transform_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Function f, Predicate pred, Iterator3 result2)
{
  *result2 = thrust::transform_if(exec, first, last, result1, f, pred);
}


template<typename ExecutionPolicy>
void TestTransformIfUnaryNoStencilDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input(3);
  Vector output(3);
  Vector result(3);
  
  input[0]   =  0; input[1]   = -2; input[2]   =  0;
  output[0]  = -1; output[1]  = -2; output[2]  = -3; 
  result[0]  = -1; result[1]  =  2; result[2]  = -3;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  transform_if_kernel<<<1,1>>>(exec,
                               input.begin(), input.end(),
                               output.begin(),
                               thrust::negate<T>(),
                               thrust::identity<T>(),
                               iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];
  
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformIfUnaryNoStencilDeviceSeq()
{
  TestTransformIfUnaryNoStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformIfUnaryNoStencilDeviceSeq);

void TestTransformIfUnaryNoStencilDeviceDevice()
{
  TestTransformIfUnaryNoStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformIfUnaryNoStencilDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Function, typename Predicate, typename Iterator4>
__global__
void transform_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 result1, Function f, Predicate pred, Iterator4 result2)
{
  *result2 = thrust::transform_if(exec, first, last, stencil_first, result1, f, pred);
}


template<typename ExecutionPolicy>
void TestTransformIfUnaryDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input(3);
  Vector stencil(3);
  Vector output(3);
  Vector result(3);
  
  input[0]   =  1; input[1]   = -2; input[2]   =  3;
  output[0]  =  1; output[1]  =  2; output[2]  =  3; 
  stencil[0] =  1; stencil[1] =  0; stencil[2] =  1;
  result[0]  = -1; result[1]  =  2; result[2]  = -3;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  transform_if_kernel<<<1,1>>>(exec,
                               input.begin(), input.end(),
                               stencil.begin(),
                               output.begin(),
                               thrust::negate<T>(),
                               thrust::identity<T>(),
                               iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];
  
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformIfUnaryDeviceSeq()
{
  TestTransformIfUnaryDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformIfUnaryDeviceSeq);

void TestTransformIfUnaryDeviceDevice()
{
  TestTransformIfUnaryDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformIfUnaryDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Function, typename Iterator4>
__global__
void transform_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 result1, Function f, Iterator4 result2)
{
  *result2 = thrust::transform(exec, first1, last1, first2, result1, f);
}


template<typename ExecutionPolicy>
void TestTransformBinaryDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input1(3);
  Vector input2(3);
  Vector output(3);
  Vector result(3);
  input1[0] =  1; input1[1] = -2; input1[2] =  3;
  input2[0] = -4; input2[1] =  5; input2[2] =  6;
  result[0] =  5; result[1] = -7; result[2] = -3;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  transform_kernel<<<1,1>>>(exec, input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minus<T>(), iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];
  
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input1.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformBinaryDeviceSeq()
{
  TestTransformBinaryDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformBinaryDeviceSeq);

void TestTransformBinaryDeviceDevice()
{
  TestTransformBinaryDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformBinaryDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Function, typename Predicate, typename Iterator5>
__global__
void transform_if_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 stencil_first, Iterator4 result1, Function f, Predicate pred, Iterator5 result2)
{
  *result2 = thrust::transform_if(exec, first1, last1, first2, stencil_first, result1, f, pred);
}


template<typename ExecutionPolicy>
void TestTransformIfBinaryDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef typename Vector::value_type T;
  
  typename Vector::iterator iter;
  
  Vector input1(3);
  Vector input2(3);
  Vector stencil(3);
  Vector output(3);
  Vector result(3);
  
  input1[0]  =  1; input1[1]  = -2; input1[2]  =  3;
  input2[0]  = -4; input2[1]  =  5; input2[2]  =  6;
  stencil[0] =  0; stencil[1] =  1; stencil[2] =  0;
  output[0]  =  1; output[1]  =  2; output[2]  =  3;
  result[0]  =  5; result[1]  =  2; result[2]  = -3;
  
  thrust::identity<T> identity;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);
  
  transform_if_kernel<<<1,1>>>(exec,
                               input1.begin(), input1.end(),
                               input2.begin(),
                               stencil.begin(),
                               output.begin(),
                               thrust::minus<T>(),
                               thrust::not1(identity),
                               iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];
  
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input1.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformIfBinaryDeviceSeq()
{
  TestTransformIfBinaryDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformIfBinaryDeviceSeq);

void TestTransformIfBinaryDeviceDevice()
{
  TestTransformIfBinaryDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformIfBinaryDeviceDevice);

void TestTransformUnaryCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector::iterator iter;

  Vector input(3);
  Vector output(3);
  Vector result(3);
  input[0]  =  1; input[1]  = -2; input[2]  =  3;
  result[0] = -1; result[1] =  2; result[2] = -3;

  cudaStream_t s;
  cudaStreamCreate(&s);

  iter = thrust::transform(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), thrust::negate<T>());
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(output, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestTransformUnaryCudaStreams);


void TestTransformBinaryCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector::iterator iter;

  Vector input1(3);
  Vector input2(3);
  Vector output(3);
  Vector result(3);
  input1[0] =  1; input1[1] = -2; input1[2] =  3;
  input2[0] = -4; input2[1] =  5; input2[2] =  6;
  result[0] =  5; result[1] = -7; result[2] = -3;

  cudaStream_t s;
  cudaStreamCreate(&s);

  iter = thrust::transform(thrust::cuda::par.on(s), input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minus<T>());
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input1.size());
  ASSERT_EQUAL(output, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestTransformBinaryCudaStreams);

