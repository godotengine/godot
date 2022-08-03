#include <unittest/unittest.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>


template <typename T>
struct less_than_five
{
  __host__ __device__ bool operator()(const T &val) const {return val < 5;}
};


template<typename ExecutionPolicy, typename Iterator, typename T1, typename T2>
__global__
void replace_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T1 old_value, T2 new_value)
{
  thrust::replace(exec, first, last, old_value, new_value);
}


template<typename T, typename ExecutionPolicy>
void TestReplaceDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  T old_value = 0;
  T new_value = 1;
  
  thrust::replace(h_data.begin(), h_data.end(), old_value, new_value);

  replace_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), old_value, new_value);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
}


template<typename T>
void TestReplaceDeviceSeq(const size_t n)
{
  TestReplaceDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceDeviceSeq);

template<typename T>
void TestReplaceDeviceDevice(const size_t n)
{
  TestReplaceDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T1, typename T2>
__global__
void replace_copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, T1 old_value, T2 new_value)
{
  thrust::replace_copy(exec, first, last, result, old_value, new_value);
}


template<typename ExecutionPolicy>
void TestReplaceCopyDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  int old_value = 0;
  int new_value = 1;
  
  thrust::host_vector<int>   h_dest(n);
  thrust::device_vector<int> d_dest(n);
  
  thrust::replace_copy(h_data.begin(), h_data.end(), h_dest.begin(), old_value, new_value);

  replace_copy_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_dest.begin(), old_value, new_value);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}

void TestReplaceCopyDeviceSeq()
{
  TestReplaceCopyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReplaceCopyDeviceSeq);

void TestReplaceCopyDeviceDevice()
{
  TestReplaceCopyDevice(thrust::device);
}
DECLARE_UNITTEST(TestReplaceCopyDeviceDevice);


template<typename ExecutionPolicy, typename Iterator, typename Predicate, typename T>
__global__
void replace_if_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, T new_value)
{
  thrust::replace_if(exec, first, last, pred, new_value);
}


template<typename ExecutionPolicy>
void TestReplaceIfDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  thrust::replace_if(h_data.begin(), h_data.end(), less_than_five<int>(), 0);

  replace_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), less_than_five<int>(), 0);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
}

void TestReplaceIfDeviceSeq()
{
  TestReplaceIfDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReplaceIfDeviceSeq);

void TestReplaceIfDeviceDevice()
{
  TestReplaceIfDevice(thrust::device);
}
DECLARE_UNITTEST(TestReplaceIfDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename T>
__global__
void replace_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, T new_value)
{
  thrust::replace_if(exec, first, last, stencil_first, pred, new_value);
}


template<typename ExecutionPolicy>
void TestReplaceIfStencilDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  thrust::host_vector<int>   h_stencil = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_stencil = h_stencil;
  
  thrust::replace_if(h_data.begin(), h_data.end(), h_stencil.begin(), less_than_five<int>(), 0);

  replace_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_stencil.begin(), less_than_five<int>(), 0);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
}

void TestReplaceIfStencilDeviceSeq()
{
  TestReplaceIfStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReplaceIfStencilDeviceSeq);

void TestReplaceIfStencilDeviceDevice()
{
  TestReplaceIfStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestReplaceIfStencilDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename T>
__global__
void replace_copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, Predicate pred, T new_value)
{
  thrust::replace_copy_if(exec, first, last, result, pred, new_value);
}


template<typename ExecutionPolicy>
void TestReplaceCopyIfDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  thrust::host_vector<int>   h_dest(n);
  thrust::device_vector<int> d_dest(n);
  
  thrust::replace_copy_if(h_data.begin(), h_data.end(), h_dest.begin(), less_than_five<int>(), 0);

  replace_copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_dest.begin(), less_than_five<int>(), 0);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}

void TestReplaceCopyIfDeviceSeq()
{
  TestReplaceCopyIfDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReplaceCopyIfDeviceSeq);

void TestReplaceCopyIfDeviceDevice()
{
  TestReplaceCopyIfDevice(thrust::device);
}
DECLARE_UNITTEST(TestReplaceCopyIfDeviceDevice);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Predicate, typename T>
__global__
void replace_copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Iterator3 result, Predicate pred, T new_value)
{
  thrust::replace_copy_if(exec, first, last, stencil_first, result, pred, new_value);
}


template<typename ExecutionPolicy>
void TestReplaceCopyIfStencilDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int>   h_data = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  thrust::host_vector<int>   h_stencil = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_stencil = h_stencil;
  
  thrust::host_vector<int>   h_dest(n);
  thrust::device_vector<int> d_dest(n);
  
  thrust::replace_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_dest.begin(), less_than_five<int>(), 0);

  replace_copy_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), d_stencil.begin(), d_dest.begin(), less_than_five<int>(), 0);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_ALMOST_EQUAL(h_data, d_data);
  ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}


void TestReplaceCopyIfStencilDeviceSeq()
{
  TestReplaceCopyIfStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReplaceCopyIfStencilDeviceSeq);


void TestReplaceCopyIfStencilDeviceDevice()
{
  TestReplaceCopyIfStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestReplaceCopyIfStencilDeviceDevice);


void TestReplaceCudaStreams()
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

  thrust::replace(thrust::cuda::par.on(s), data.begin(), data.end(), (T) 1, (T) 4);
  thrust::replace(thrust::cuda::par.on(s), data.begin(), data.end(), (T) 2, (T) 5);

  cudaStreamSynchronize(s);

  Vector result(5);
  result[0] =  4; 
  result[1] =  5; 
  result[2] =  4;
  result[3] =  3; 
  result[4] =  5; 

  ASSERT_EQUAL(data, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestReplaceCudaStreams);

