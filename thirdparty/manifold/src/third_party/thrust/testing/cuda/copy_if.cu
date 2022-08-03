#include <unittest/unittest.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>


template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) { return (static_cast<unsigned int>(x) & 1) == 0; }
};


template<typename T>
struct mod_3
{
  __host__ __device__
  unsigned int operator()(T x) { return static_cast<unsigned int>(x) % 3; }
};


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__ void copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Predicate pred, Iterator3 result2)
{
  *result2 = thrust::copy_if(exec, first, last, result1, pred);
}


template<typename ExecutionPolicy>
void TestCopyIfDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  typename thrust::host_vector<int>::iterator   h_new_end;
  typename thrust::device_vector<int>::iterator d_new_end;

  thrust::device_vector<
    typename thrust::device_vector<int>::iterator
  > d_new_end_vec(1);
  
  // test with Predicate that returns a bool
  {
    thrust::host_vector<int>   h_result(n);
    thrust::device_vector<int> d_result(n);
    
    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());

    copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), is_even<int>(), d_new_end_vec.begin());
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);

    d_new_end = d_new_end_vec[0];
    
    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());
    
    ASSERT_EQUAL(h_result, d_result);
  }
  
  // test with Predicate that returns a non-bool
  {
    thrust::host_vector<int>   h_result(n);
    thrust::device_vector<int> d_result(n);
    
    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<int>());

    copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), mod_3<int>(), d_new_end_vec.begin());
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);

    d_new_end = d_new_end_vec[0];
    
    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());
    
    ASSERT_EQUAL(h_result, d_result);
  }
}


void TestCopyIfDeviceSeq()
{
  TestCopyIfDevice(thrust::seq);
}
DECLARE_UNITTEST(TestCopyIfDeviceSeq);


void TestCopyIfDeviceDevice()
{
  TestCopyIfDevice(thrust::device);
}
DECLARE_UNITTEST(TestCopyIfDeviceDevice);


void TestCopyIfDeviceNoSync()
{
  TestCopyIfDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestCopyIfDeviceNoSync);

template<typename ExecutionPolicy>
void TestCopyIfCudaStreams(ExecutionPolicy policy)
{
  typedef thrust::device_vector<int> Vector;

  Vector data(5);
  data[0] =  1; 
  data[1] =  2; 
  data[2] =  1;
  data[3] =  3; 
  data[4] =  2; 

  Vector result(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Vector::iterator end = thrust::copy_if(policy.on(s),
                                         data.begin(), 
                                         data.end(), 
                                         result.begin(),
                                         is_even<int>());

  ASSERT_EQUAL(end - result.begin(), 2);

  ASSERT_EQUAL(result[0], 2);
  ASSERT_EQUAL(result[1], 2);

  cudaStreamDestroy(s);
}

void TestCopyIfCudaStreamsSync(){
  TestCopyIfCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestCopyIfCudaStreamsSync);

void TestCopyIfCudaStreamsNoSync(){
  TestCopyIfCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestCopyIfCudaStreamsNoSync);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Predicate, typename Iterator4>
__global__ void copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 result1, Predicate pred, Iterator4 result2)
{
  *result2 = thrust::copy_if(exec, first, last, stencil_first, result1, pred);
}


template<typename ExecutionPolicy>
void TestCopyIfStencilDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data(n); thrust::sequence(h_data.begin(), h_data.end());
  thrust::device_vector<int> d_data(n); thrust::sequence(d_data.begin(), d_data.end()); 
  
  thrust::host_vector<int>   h_stencil = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_stencil = unittest::random_integers<int>(n);
  
  typename thrust::host_vector<int>::iterator   h_new_end;
  typename thrust::device_vector<int>::iterator d_new_end;

  thrust::device_vector<
    typename thrust::device_vector<int>::iterator
  > d_new_end_vec(1);
  
  // test with Predicate that returns a bool
  {
    thrust::host_vector<int>   h_result(n);
    thrust::device_vector<int> d_result(n);
    
    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());

    copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), is_even<int>(), d_new_end_vec.begin());
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);

    d_new_end = d_new_end_vec[0];
    
    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());
    
    ASSERT_EQUAL(h_result, d_result);
  }
  
  // test with Predicate that returns a non-bool
  {
    thrust::host_vector<int>   h_result(n);
    thrust::device_vector<int> d_result(n);
    
    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<int>());

    copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), mod_3<int>(), d_new_end_vec.begin());
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);

    d_new_end = d_new_end_vec[0];
    
    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());
    
    ASSERT_EQUAL(h_result, d_result);
  }
}


void TestCopyIfStencilDeviceSeq()
{
  TestCopyIfStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestCopyIfStencilDeviceSeq);


void TestCopyIfStencilDeviceDevice()
{
  TestCopyIfStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestCopyIfStencilDeviceDevice);


void TestCopyIfStencilDeviceNoSync()
{
  TestCopyIfStencilDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestCopyIfStencilDeviceNoSync);


template<typename ExecutionPolicy>
void TestCopyIfStencilCudaStreams(ExecutionPolicy policy)
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

  Vector stencil(5);
  stencil[0] = 0;
  stencil[1] = 1;
  stencil[2] = 0;
  stencil[3] = 0;
  stencil[4] = 1;

  cudaStream_t s;
  cudaStreamCreate(&s);

  Vector::iterator end = thrust::copy_if(policy.on(s),
                                         data.begin(), 
                                         data.end(),
                                         stencil.begin(),
                                         result.begin(),
                                         thrust::identity<T>());

  ASSERT_EQUAL(end - result.begin(), 2);

  ASSERT_EQUAL(result[0], 2);
  ASSERT_EQUAL(result[1], 2);

  cudaStreamDestroy(s);
}

void TestCopyIfStencilCudaStreamsSync()
{
  TestCopyIfStencilCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestCopyIfStencilCudaStreamsSync);


void TestCopyIfStencilCudaStreamsNoSync()
{
  TestCopyIfStencilCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestCopyIfStencilCudaStreamsNoSync);

