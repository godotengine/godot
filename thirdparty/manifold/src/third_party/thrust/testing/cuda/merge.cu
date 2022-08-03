#include <unittest/unittest.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4>
__global__
void merge_kernel(ExecutionPolicy exec,
                  Iterator1 first1, Iterator1 last1,
                  Iterator2 first2, Iterator2 last2,
                  Iterator3 result1,
                  Iterator4 result2)
{
  *result2 = thrust::merge(exec, first1, last1, first2, last2, result1);
}


template<typename ExecutionPolicy>
void TestMergeDevice(ExecutionPolicy exec)
{
  size_t n = 10000;
  size_t sizes[]   = {0, 1, n / 2, n, n + 1, 2 * n};
  size_t num_sizes = sizeof(sizes) / sizeof(size_t);

  thrust::host_vector<int> random = unittest::random_integers<unittest::int8_t>(n + *thrust::max_element(sizes, sizes + num_sizes));

  thrust::host_vector<int> h_a(random.begin(), random.begin() + n);
  thrust::host_vector<int> h_b(random.begin() + n, random.end());

  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());
  
  thrust::device_vector<int> d_a = h_a;
  thrust::device_vector<int> d_b = h_b;

  for(size_t i = 0; i < num_sizes; i++)
  {
    size_t size = sizes[i];
    
    thrust::host_vector<int>   h_result(n + size);
    thrust::device_vector<int> d_result(n + size);

    typename thrust::host_vector<int>::iterator   h_end;

    typedef typename thrust::device_vector<int>::iterator iter_type;
    thrust::device_vector<iter_type> d_end(1);
    
    h_end = thrust::merge(h_a.begin(), h_a.end(),
                          h_b.begin(), h_b.begin() + size,
                          h_result.begin());
    h_result.resize(h_end - h_result.begin());

    merge_kernel<<<1,1>>>(exec,
                          d_a.begin(), d_a.end(),
                          d_b.begin(), d_b.begin() + size,
                          d_result.begin(),
                          d_end.begin());
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);

    d_result.resize((iter_type)d_end[0] - d_result.begin());

    ASSERT_EQUAL(h_result, d_result);
  }
}


void TestMergeDeviceSeq()
{
  TestMergeDevice(thrust::seq);
}
DECLARE_UNITTEST(TestMergeDeviceSeq);


void TestMergeDeviceDevice()
{
  TestMergeDevice(thrust::device);
}
DECLARE_UNITTEST(TestMergeDeviceDevice);


void TestMergeCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::iterator Iterator;

  Vector a(3), b(4);

  a[0] = 0; a[1] = 2; a[2] = 4;
  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;

  Vector ref(7);
  ref[0] = 0;
  ref[1] = 0;
  ref[2] = 2;
  ref[3] = 3;
  ref[4] = 3;
  ref[5] = 4;
  ref[6] = 4;

  Vector result(7);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Iterator end = thrust::merge(thrust::cuda::par.on(s),
                               a.begin(), a.end(),
                               b.begin(), b.end(),
                               result.begin());

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestMergeCudaStreams);

