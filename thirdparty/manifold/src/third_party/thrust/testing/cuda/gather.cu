#include <unittest/unittest.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <algorithm>

template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void gather_kernel(ExecutionPolicy exec, Iterator1 map_first, Iterator1 map_last, Iterator2 elements_first, Iterator3 result)
{
  thrust::gather(exec, map_first, map_last, elements_first, result);
}


template<typename T, typename ExecutionPolicy>
void TestGatherDevice(ExecutionPolicy exec, const size_t n)
{
  const size_t source_size = std::min((size_t) 10, 2 * n);
  
  // source vectors to gather from
  thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
  thrust::device_vector<T> d_source = h_source;
  
  // gather indices
  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);
  
  for(size_t i = 0; i < n; i++)
    h_map[i] =  h_map[i] % source_size;
  
  thrust::device_vector<unsigned int> d_map = h_map;
  
  // gather destination
  thrust::host_vector<T>   h_output(n);
  thrust::device_vector<T> d_output(n);
  
  thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), h_output.begin());

  gather_kernel<<<1,1>>>(exec, d_map.begin(), d_map.end(), d_source.begin(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_output, d_output);
}

template<typename T>
void TestGatherDeviceSeq(const size_t n)
{
  TestGatherDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestGatherDeviceSeq);

template<typename T>
void TestGatherDeviceDevice(const size_t n)
{
  TestGatherDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestGatherDeviceDevice);


void TestGatherCudaStreams()
{
  thrust::device_vector<int> map(5);  // gather indices
  thrust::device_vector<int> src(8);  // source vector
  thrust::device_vector<int> dst(5);  // destination vector
  
  map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
  src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
  dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::gather(thrust::cuda::par.on(s), map.begin(), map.end(), src.begin(), dst.begin());
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(dst[0], 6);
  ASSERT_EQUAL(dst[1], 2);
  ASSERT_EQUAL(dst[2], 1);
  ASSERT_EQUAL(dst[3], 7);
  ASSERT_EQUAL(dst[4], 2);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestGatherCudaStreams);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Predicate>
__global__
void gather_if_kernel(ExecutionPolicy exec, Iterator1 map_first, Iterator1 map_last, Iterator2 stencil_first, Iterator3 elements_first, Iterator4 result, Predicate pred)
{
  thrust::gather_if(exec, map_first, map_last, stencil_first, elements_first, result, pred);
}


template<typename T>
struct is_even_gather_if
{
  __host__ __device__
  bool operator()(const T i) const
  { 
    return (i % 2) == 0;
  }
};


template<typename T, typename ExecutionPolicy>
void TestGatherIfDevice(ExecutionPolicy exec, const size_t n)
{
  const size_t source_size = std::min((size_t) 10, 2 * n);
  
  // source vectors to gather from
  thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
  thrust::device_vector<T> d_source = h_source;
  
  // gather indices
  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);
  
  for(size_t i = 0; i < n; i++)
      h_map[i] = h_map[i] % source_size;
  
  thrust::device_vector<unsigned int> d_map = h_map;
  
  // gather stencil
  thrust::host_vector<unsigned int> h_stencil = unittest::random_integers<unsigned int>(n);
  
  for(size_t i = 0; i < n; i++)
    h_stencil[i] = h_stencil[i] % 2;
  
  thrust::device_vector<unsigned int> d_stencil = h_stencil;
  
  // gather destination
  thrust::host_vector<T>   h_output(n);
  thrust::device_vector<T> d_output(n);
  
  thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_source.begin(), h_output.begin(), is_even_gather_if<unsigned int>());

  gather_if_kernel<<<1,1>>>(exec, d_map.begin(), d_map.end(), d_stencil.begin(), d_source.begin(), d_output.begin(), is_even_gather_if<unsigned int>());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(h_output, d_output);
}

template<typename T>
void TestGatherIfDeviceSeq(const size_t n)
{
  TestGatherIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestGatherIfDeviceSeq);

template<typename T>
void TestGatherIfDeviceDevice(const size_t n)
{
  TestGatherIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestGatherIfDeviceDevice);

void TestGatherIfCudaStreams(void)
{
  thrust::device_vector<int> flg(5);  // predicate array
  thrust::device_vector<int> map(5);  // gather indices
  thrust::device_vector<int> src(8);  // source vector
  thrust::device_vector<int> dst(5);  // destination vector
  
  flg[0] = 0; flg[1] = 1; flg[2] = 0; flg[3] = 1; flg[4] = 0;
  map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
  src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
  dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

  cudaStream_t s;
  cudaStreamCreate(&s);
  
  thrust::gather_if(thrust::cuda::par.on(s), map.begin(), map.end(), flg.begin(), src.begin(), dst.begin());
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(dst[0], 0);
  ASSERT_EQUAL(dst[1], 2);
  ASSERT_EQUAL(dst[2], 0);
  ASSERT_EQUAL(dst[3], 7);
  ASSERT_EQUAL(dst[4], 0);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestGatherIfCudaStreams);

