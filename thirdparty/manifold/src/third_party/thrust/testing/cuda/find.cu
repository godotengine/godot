#include <unittest/unittest.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>


template<typename T>
struct equal_to_value_pred
{
    T value;

    equal_to_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v == value; }
};


template<typename T>
struct not_equal_to_value_pred
{
    T value;

    not_equal_to_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v != value; }
};


template<typename T>
struct less_than_value_pred
{
    T value;

    less_than_value_pred(T value) : value(value) {}

    __host__ __device__
    bool operator()(T v) const { return v < value; }
};


template<typename ExecutionPolicy, typename Iterator, typename T, typename Iterator2>
__global__ void find_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T value, Iterator2 result)
{
  *result = thrust::find(exec, first, last, value);
}


template<typename ExecutionPolicy>
void TestFindDevice(ExecutionPolicy exec)
{
  size_t n = 100;

  thrust::host_vector<int>   h_data = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  typename thrust::host_vector<int>::iterator   h_iter;
  
  typedef typename thrust::device_vector<int>::iterator iter_type;
  thrust::device_vector<iter_type> d_result(1);
  
  h_iter = thrust::find(h_data.begin(), h_data.end(), int(0));

  find_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), int(0), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
  
  for(size_t i = 1; i < n; i *= 2)
  {
    int sample = h_data[i];

    h_iter = thrust::find(h_data.begin(), h_data.end(), sample);

    find_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), sample, d_result.begin());
    {
      cudaError_t const err = cudaDeviceSynchronize();
      ASSERT_EQUAL(cudaSuccess, err);
    }

    ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
  }
}


void TestFindDeviceSeq()
{
  TestFindDevice(thrust::seq);
};
DECLARE_UNITTEST(TestFindDeviceSeq);


void TestFindDeviceDevice()
{
  TestFindDevice(thrust::device);
};
DECLARE_UNITTEST(TestFindDeviceDevice);


template<typename ExecutionPolicy, typename Iterator, typename Predicate, typename Iterator2>
__global__ void find_if_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::find_if(exec, first, last, pred);
}


template<typename ExecutionPolicy>
void TestFindIfDevice(ExecutionPolicy exec)
{
  size_t n = 100;

  thrust::host_vector<int>   h_data = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  typename thrust::host_vector<int>::iterator   h_iter;
  
  typedef typename thrust::device_vector<int>::iterator iter_type;
  thrust::device_vector<iter_type> d_result(1);
  
  h_iter = thrust::find_if(h_data.begin(), h_data.end(), equal_to_value_pred<int>(0));

  find_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), equal_to_value_pred<int>(0), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
  
  for (size_t i = 1; i < n; i *= 2)
  {
    int sample = h_data[i];

    h_iter = thrust::find_if(h_data.begin(), h_data.end(), equal_to_value_pred<int>(sample));

    find_if_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), equal_to_value_pred<int>(sample), d_result.begin());
    {
      cudaError_t const err = cudaDeviceSynchronize();
      ASSERT_EQUAL(cudaSuccess, err);
    }

    ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
  }
}


void TestFindIfDeviceSeq()
{
  TestFindIfDevice(thrust::seq);
};
DECLARE_UNITTEST(TestFindIfDeviceSeq);


void TestFindIfDeviceDevice()
{
  TestFindIfDevice(thrust::device);
};
DECLARE_UNITTEST(TestFindIfDeviceDevice);


template<typename ExecutionPolicy, typename Iterator, typename Predicate, typename Iterator2>
__global__ void find_if_not_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::find_if_not(exec, first, last, pred);
}


template<typename ExecutionPolicy>
void TestFindIfNotDevice(ExecutionPolicy exec)
{
  size_t n = 100;
  thrust::host_vector<int>   h_data = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_data = h_data;
  
  typename thrust::host_vector<int>::iterator   h_iter;
  
  typedef typename thrust::device_vector<int>::iterator iter_type;
  thrust::device_vector<iter_type> d_result(1);
  
  h_iter = thrust::find_if_not(h_data.begin(), h_data.end(), not_equal_to_value_pred<int>(0));

  find_if_not_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), not_equal_to_value_pred<int>(0), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
  
  for(size_t i = 1; i < n; i *= 2)
  {
    int sample = h_data[i];

    h_iter = thrust::find_if_not(h_data.begin(), h_data.end(), not_equal_to_value_pred<int>(sample));

    find_if_not_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), not_equal_to_value_pred<int>(sample), d_result.begin());
    {
      cudaError_t const err = cudaDeviceSynchronize();
      ASSERT_EQUAL(cudaSuccess, err);
    }

    ASSERT_EQUAL(h_iter - h_data.begin(), (iter_type)d_result[0] - d_data.begin());
  }
}


void TestFindIfNotDeviceSeq()
{
  TestFindIfNotDevice(thrust::seq);
};
DECLARE_UNITTEST(TestFindIfNotDeviceSeq);


void TestFindIfNotDeviceDevice()
{
  TestFindIfNotDevice(thrust::device);
};
DECLARE_UNITTEST(TestFindIfNotDeviceDevice);


void TestFindCudaStreams()
{
  thrust::device_vector<int> vec(5);
  vec[0] = 1;
  vec[1] = 2;
  vec[2] = 3;
  vec[3] = 3;
  vec[4] = 5;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  ASSERT_EQUAL(thrust::find(thrust::cuda::par.on(s), vec.begin(), vec.end(), 0) - vec.begin(), 5);
  ASSERT_EQUAL(thrust::find(thrust::cuda::par.on(s), vec.begin(), vec.end(), 1) - vec.begin(), 0);
  ASSERT_EQUAL(thrust::find(thrust::cuda::par.on(s), vec.begin(), vec.end(), 2) - vec.begin(), 1);
  ASSERT_EQUAL(thrust::find(thrust::cuda::par.on(s), vec.begin(), vec.end(), 3) - vec.begin(), 2);
  ASSERT_EQUAL(thrust::find(thrust::cuda::par.on(s), vec.begin(), vec.end(), 4) - vec.begin(), 5);
  ASSERT_EQUAL(thrust::find(thrust::cuda::par.on(s), vec.begin(), vec.end(), 5) - vec.begin(), 4);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestFindCudaStreams);

