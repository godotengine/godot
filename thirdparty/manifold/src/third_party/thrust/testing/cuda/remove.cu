#include <unittest/unittest.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename T, typename Iterator2>
__global__
void remove_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T val, Iterator2 result)
{
  *result = thrust::remove(exec, first, last, val);
}


template<typename ExecutionPolicy, typename Iterator, typename Predicate, typename Iterator2>
__global__
void remove_if_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::remove_if(exec, first, last, pred);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__
void remove_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, Iterator3 result)
{
  *result = thrust::remove_if(exec, first, last, stencil_first, pred);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T, typename Iterator3>
__global__
void remove_copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, T val, Iterator3 result2)
{
  *result2 = thrust::remove_copy(exec, first, last, result1, val);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__
void remove_copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, Predicate pred, Iterator3 result_end)
{
  *result_end = thrust::remove_copy_if(exec, first, last, result, pred);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Predicate, typename Iterator4>
__global__
void remove_copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 result, Predicate pred, Iterator4 result_end)
{
  *result_end = thrust::remove_copy_if(exec, first, last, stencil_first, result, pred);
}


template<typename T>
struct is_even
  : thrust::unary_function<T,bool>
{
  __host__ __device__
  bool operator()(T x) { return (static_cast<unsigned int>(x) & 1) == 0; }
};


template<typename T>
struct is_true
  : thrust::unary_function<T,bool>
{
  __host__ __device__
  bool operator()(T x) { return x ? true : false; }
};


template<typename ExecutionPolicy>
void TestRemoveDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  typedef typename thrust::device_vector<int>::iterator iterator;
  thrust::device_vector<iterator> d_result(1);
  
  size_t h_size = thrust::remove(h_data.begin(), h_data.end(), 0) - h_data.begin();

  remove_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), 0, d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  size_t d_size = (iterator)d_result[0] - d_data.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_data.resize(h_size);
  d_data.resize(d_size);
  
  ASSERT_EQUAL(h_data, d_data);
}


void TestRemoveDeviceSeq()
{
  TestRemoveDevice(thrust::seq);
}
DECLARE_UNITTEST(TestRemoveDeviceSeq);


void TestRemoveDeviceDevice()
{
  TestRemoveDevice(thrust::device);
}
DECLARE_UNITTEST(TestRemoveDeviceDevice);


template<typename ExecutionPolicy>
void TestRemoveIfDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  typedef typename thrust::device_vector<int>::iterator iterator;
  thrust::device_vector<iterator> d_result(1);
  
  size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), is_true<int>()) - h_data.begin();

  remove_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), is_true<int>(), d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  size_t d_size = (iterator)d_result[0] - d_data.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_data.resize(h_size);
  d_data.resize(d_size);
  
  ASSERT_EQUAL(h_data, d_data);
}


void TestRemoveIfDeviceSeq()
{
  TestRemoveIfDevice(thrust::seq);
}
DECLARE_UNITTEST(TestRemoveIfDeviceSeq);


void TestRemoveIfDeviceDevice()
{
  TestRemoveIfDevice(thrust::device);
}
DECLARE_UNITTEST(TestRemoveIfDeviceDevice);


template<typename ExecutionPolicy>
void TestRemoveIfStencilDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  typedef typename thrust::device_vector<int>::iterator iterator;
  thrust::device_vector<iterator> d_result(1);
  
  thrust::host_vector<bool>   h_stencil = unittest::random_integers<bool>(n);
  thrust::device_vector<bool> d_stencil = h_stencil;
  
  size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), h_stencil.begin(), is_true<int>()) - h_data.begin();

  remove_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_stencil.begin(), is_true<int>(), d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  size_t d_size = (iterator)d_result[0] - d_data.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_data.resize(h_size);
  d_data.resize(d_size);
  
  ASSERT_EQUAL(h_data, d_data);
}


void TestRemoveIfStencilDeviceSeq()
{
  TestRemoveIfStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestRemoveIfStencilDeviceSeq);


void TestRemoveIfStencilDeviceDevice()
{
  TestRemoveIfStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestRemoveIfStencilDeviceDevice);


template<typename ExecutionPolicy>
void TestRemoveCopyDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  thrust::host_vector<int>   h_result(n);
  thrust::device_vector<int> d_result(n);

  typedef typename thrust::device_vector<int>::iterator iterator;
  thrust::device_vector<iterator> d_new_end(1);
  
  size_t h_size = thrust::remove_copy(h_data.begin(), h_data.end(), h_result.begin(), 0) - h_result.begin();

  remove_copy_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), 0, d_new_end.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  size_t d_size = (iterator)d_new_end[0] - d_result.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_result.resize(h_size);
  d_result.resize(d_size);
  
  ASSERT_EQUAL(h_result, d_result);
}


void TestRemoveCopyDeviceSeq()
{
  TestRemoveCopyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestRemoveCopyDeviceSeq);


void TestRemoveCopyDeviceDevice()
{
  TestRemoveCopyDevice(thrust::device);
}
DECLARE_UNITTEST(TestRemoveCopyDeviceDevice);


template<typename ExecutionPolicy>
void TestRemoveCopyIfDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  thrust::host_vector<int>   h_result(n);
  thrust::device_vector<int> d_result(n);

  typedef typename thrust::device_vector<int>::iterator iterator;
  thrust::device_vector<iterator> d_new_end(1);
  
  size_t h_size = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_true<int>()) - h_result.begin();

  remove_copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), is_true<int>(), d_new_end.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  size_t d_size = (iterator)d_new_end[0] - d_result.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_result.resize(h_size);
  d_result.resize(d_size);
  
  ASSERT_EQUAL(h_result, d_result);
}


void TestRemoveCopyIfDeviceSeq()
{
  TestRemoveCopyIfDevice(thrust::seq);
}
DECLARE_UNITTEST(TestRemoveCopyIfDeviceSeq);


void TestRemoveCopyIfDeviceDevice()
{
  TestRemoveCopyIfDevice(thrust::device);
}
DECLARE_UNITTEST(TestRemoveCopyIfDeviceDevice);


template<typename ExecutionPolicy>
void TestRemoveCopyIfStencilDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  thrust::host_vector<int>   h_result(n);
  thrust::device_vector<int> d_result(n);

  typedef typename thrust::device_vector<int>::iterator iterator;
  thrust::device_vector<iterator> d_new_end(1);

  thrust::host_vector<bool>   h_stencil = unittest::random_integers<bool>(n);
  thrust::device_vector<bool> d_stencil = h_stencil;
  
  size_t h_size = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), is_true<int>()) - h_result.begin();

  remove_copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), is_true<int>(), d_new_end.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  size_t d_size = (iterator)d_new_end[0] - d_result.begin();
  
  ASSERT_EQUAL(h_size, d_size);
  
  h_result.resize(h_size);
  d_result.resize(d_size);
  
  ASSERT_EQUAL(h_result, d_result);
}


void TestRemoveCopyIfStencilDeviceSeq()
{
  TestRemoveCopyIfStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestRemoveCopyIfStencilDeviceSeq);


void TestRemoveCopyIfStencilDeviceDevice()
{
  TestRemoveCopyIfStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestRemoveCopyIfStencilDeviceDevice);


void TestRemoveCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  3; 
  data[4] =  2; 

  cudaStream_t s;
  cudaStreamCreate(&s);

  Vector::iterator end = thrust::remove(thrust::cuda::par.on(s),
                                        data.begin(), 
                                        data.end(), 
                                        (T) 2);

  ASSERT_EQUAL(end - data.begin(), 3);

  ASSERT_EQUAL(data[0], 1);
  ASSERT_EQUAL(data[1], 1);
  ASSERT_EQUAL(data[2], 3);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestRemoveCudaStreams);


void TestRemoveCopyCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  3; 
  data[4] =  2; 

  Vector result(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Vector::iterator end = thrust::remove_copy(thrust::cuda::par.on(s),
                                             data.begin(), 
                                             data.end(), 
                                             result.begin(), 
                                             (T) 2);

  ASSERT_EQUAL(end - result.begin(), 3);

  ASSERT_EQUAL(result[0], 1);
  ASSERT_EQUAL(result[1], 1);
  ASSERT_EQUAL(result[2], 3);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestRemoveCopyCudaStreams);


void TestRemoveIfCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  3; 
  data[4] =  2; 

  cudaStream_t s;
  cudaStreamCreate(&s);

  Vector::iterator end = thrust::remove_if(thrust::cuda::par.on(s),
                                           data.begin(), 
                                           data.end(), 
                                           is_even<T>());

  ASSERT_EQUAL(end - data.begin(), 3);

  ASSERT_EQUAL(data[0], 1);
  ASSERT_EQUAL(data[1], 1);
  ASSERT_EQUAL(data[2], 3);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestRemoveIfCudaStreams);


void TestRemoveIfStencilCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  3; 
  data[4] =  2; 

  Vector stencil(5);
  stencil[0] = 0;
  stencil[1] = 1;
  stencil[2] = 0;
  stencil[3] = 0;
  stencil[4] = 1;

  cudaStream_t s;
  cudaStreamCreate(&s);

  Vector::iterator end = thrust::remove_if(thrust::cuda::par.on(s),
                                           data.begin(), 
                                           data.end(),
                                           stencil.begin(),
                                           thrust::identity<T>());

  ASSERT_EQUAL(end - data.begin(), 3);

  ASSERT_EQUAL(data[0], 1);
  ASSERT_EQUAL(data[1], 1);
  ASSERT_EQUAL(data[2], 3);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestRemoveIfStencilCudaStreams);


void TestRemoveCopyIfCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  3; 
  data[4] =  2; 

  Vector result(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Vector::iterator end = thrust::remove_copy_if(thrust::cuda::par.on(s),
                                                data.begin(), 
                                                data.end(), 
                                                result.begin(), 
                                                is_even<T>());

  ASSERT_EQUAL(end - result.begin(), 3);

  ASSERT_EQUAL(result[0], 1);
  ASSERT_EQUAL(result[1], 1);
  ASSERT_EQUAL(result[2], 3);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestRemoveCopyIfCudaStreams);


void TestRemoveCopyIfStencilCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  3; 
  data[4] =  2; 

  Vector stencil(5);
  stencil[0] = 0;
  stencil[1] = 1;
  stencil[2] = 0;
  stencil[3] = 0;
  stencil[4] = 1;

  Vector result(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Vector::iterator end = thrust::remove_copy_if(thrust::cuda::par.on(s),
                                                data.begin(), 
                                                data.end(), 
                                                stencil.begin(),
                                                result.begin(), 
                                                thrust::identity<T>());

  ASSERT_EQUAL(end - result.begin(), 3);

  ASSERT_EQUAL(result[0], 1);
  ASSERT_EQUAL(result[1], 1);
  ASSERT_EQUAL(result[2], 3);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestRemoveCopyIfStencilCudaStreams);

