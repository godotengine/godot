#include <unittest/unittest.h>
#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Predicate, typename Iterator2>
__global__
void partition_point_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Predicate pred, Iterator2 result)
{
  *result = thrust::partition_point(exec, first, last, pred);
}


template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) const { return ((int) x % 2) == 0; }
};


template<typename ExecutionPolicy>
void TestPartitionPointDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::device_vector<int> v = unittest::random_integers<int>(n);
  typedef typename thrust::device_vector<int>::iterator iterator;

  iterator ref = thrust::stable_partition(v.begin(), v.end(), is_even<int>());

  thrust::device_vector<iterator> result(1);
  partition_point_kernel<<<1,1>>>(exec, v.begin(), v.end(), is_even<int>(), result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(ref - v.begin(), (iterator)result[0] - v.begin());
}


void TestPartitionPointDeviceSeq()
{
  TestPartitionPointDevice(thrust::seq);
}
DECLARE_UNITTEST(TestPartitionPointDeviceSeq);


void TestPartitionPointDeviceDevice()
{
  TestPartitionPointDevice(thrust::device);
}
DECLARE_UNITTEST(TestPartitionPointDeviceDevice);


void TestPartitionPointCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  typedef Vector::iterator Iterator;

  Vector v(4);
  v[0] = 1; v[1] = 1; v[2] = 1; v[3] = 0;

  Iterator first = v.begin();

  Iterator last = v.begin() + 4;
  Iterator ref = first + 3;

  cudaStream_t s;
  cudaStreamCreate(&s);

  ASSERT_EQUAL_QUIET(ref, thrust::partition_point(thrust::cuda::par.on(s), first, last, thrust::identity<T>()));

  last = v.begin() + 3;
  ref = last;
  ASSERT_EQUAL_QUIET(ref, thrust::partition_point(thrust::cuda::par.on(s), first, last, thrust::identity<T>()));

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestPartitionPointCudaStreams);

