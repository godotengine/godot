#include <unittest/unittest.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


template<typename T>
struct is_equal_div_10_unique
{
  __host__ __device__
  bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
};


template<typename Vector>
void initialize_keys(Vector& keys)
{
  keys.resize(9);
  keys[0] = 11;
  keys[1] = 11;
  keys[2] = 21;
  keys[3] = 20;
  keys[4] = 21;
  keys[5] = 21;
  keys[6] = 21;
  keys[7] = 37;
  keys[8] = 37;
}


template<typename Vector>
void initialize_values(Vector& values)
{
  values.resize(9);
  values[0] = 0; 
  values[1] = 1;
  values[2] = 2;
  values[3] = 3;
  values[4] = 4;
  values[5] = 5;
  values[6] = 6;
  values[7] = 7;
  values[8] = 8;
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__
void unique_by_key_kernel(ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result)
{
  *result = thrust::unique_by_key(exec, keys_first, keys_last, values_first);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename BinaryPredicate, typename Iterator3>
__global__
void unique_by_key_kernel(ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, BinaryPredicate pred, Iterator3 result)
{
  *result = thrust::unique_by_key(exec, keys_first, keys_last, values_first, pred);
}


template<typename ExecutionPolicy>
void TestUniqueByKeyDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector keys;
  Vector values;
  
  typedef thrust::pair<typename Vector::iterator, typename Vector::iterator> iter_pair;
  thrust::device_vector<iter_pair> new_last_vec(1);
  iter_pair new_last;
  
  // basic test
  initialize_keys(keys);  initialize_values(values);
  
  unique_by_key_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - values.begin(), 5);
  ASSERT_EQUAL(keys[0], 11);
  ASSERT_EQUAL(keys[1], 21);
  ASSERT_EQUAL(keys[2], 20);
  ASSERT_EQUAL(keys[3], 21);
  ASSERT_EQUAL(keys[4], 37);
  
  ASSERT_EQUAL(values[0], 0);
  ASSERT_EQUAL(values[1], 2);
  ASSERT_EQUAL(values[2], 3);
  ASSERT_EQUAL(values[3], 4);
  ASSERT_EQUAL(values[4], 7);
  
  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  unique_by_key_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), is_equal_div_10_unique<T>(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - keys.begin(),   3);
  ASSERT_EQUAL(new_last.second - values.begin(), 3);
  ASSERT_EQUAL(keys[0], 11);
  ASSERT_EQUAL(keys[1], 21);
  ASSERT_EQUAL(keys[2], 37);
  
  ASSERT_EQUAL(values[0], 0);
  ASSERT_EQUAL(values[1], 2);
  ASSERT_EQUAL(values[2], 7);
}

void TestUniqueByKeyDeviceSeq()
{
  TestUniqueByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueByKeyDeviceSeq);


void TestUniqueByKeyDeviceDevice()
{
  TestUniqueByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueByKeyDeviceDevice);


void TestUniqueByKeyDeviceNoSync()
{
  TestUniqueByKeyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueByKeyDeviceNoSync);


template<typename ExecutionPolicy>
void TestUniqueByKeyCudaStreams(ExecutionPolicy policy)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;
  
  Vector keys;
  Vector values;
  
  typedef thrust::pair<Vector::iterator, Vector::iterator> iter_pair;
  iter_pair new_last;
  
  // basic test
  initialize_keys(keys);  initialize_values(values);

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);
  
  new_last = thrust::unique_by_key(streampolicy, keys.begin(), keys.end(), values.begin());
  cudaStreamSynchronize(s);
  
  ASSERT_EQUAL(new_last.first  - keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - values.begin(), 5);
  ASSERT_EQUAL(keys[0], 11);
  ASSERT_EQUAL(keys[1], 21);
  ASSERT_EQUAL(keys[2], 20);
  ASSERT_EQUAL(keys[3], 21);
  ASSERT_EQUAL(keys[4], 37);
  
  ASSERT_EQUAL(values[0], 0);
  ASSERT_EQUAL(values[1], 2);
  ASSERT_EQUAL(values[2], 3);
  ASSERT_EQUAL(values[3], 4);
  ASSERT_EQUAL(values[4], 7);
  
  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  new_last = thrust::unique_by_key(streampolicy, keys.begin(), keys.end(), values.begin(), is_equal_div_10_unique<T>());
  
  ASSERT_EQUAL(new_last.first  - keys.begin(),   3);
  ASSERT_EQUAL(new_last.second - values.begin(), 3);
  ASSERT_EQUAL(keys[0], 11);
  ASSERT_EQUAL(keys[1], 21);
  ASSERT_EQUAL(keys[2], 37);
  
  ASSERT_EQUAL(values[0], 0);
  ASSERT_EQUAL(values[1], 2);
  ASSERT_EQUAL(values[2], 7);

  cudaStreamDestroy(s);
}

void TestUniqueByKeyCudaStreamsSync()
{
  TestUniqueByKeyCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueByKeyCudaStreamsSync);


void TestUniqueByKeyCudaStreamsNoSync()
{
  TestUniqueByKeyCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueByKeyCudaStreamsNoSync);


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5>
__global__
void unique_by_key_copy_kernel(ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 keys_result, Iterator4 values_result, Iterator5 result)
{
  *result = thrust::unique_by_key_copy(exec, keys_first, keys_last, values_first, keys_result, values_result);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename BinaryPredicate, typename Iterator5>
__global__
void unique_by_key_copy_kernel(ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 keys_result, Iterator4 values_result, BinaryPredicate pred, Iterator5 result)
{
  *result = thrust::unique_by_key_copy(exec, keys_first, keys_last, values_first, keys_result, values_result, pred);
}


template<typename ExecutionPolicy>
void TestUniqueCopyByKeyDevice(ExecutionPolicy exec)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector keys;
  Vector values;

  typedef thrust::pair<typename Vector::iterator, typename Vector::iterator> iter_pair;
  thrust::device_vector<iter_pair> new_last_vec(1);
  iter_pair new_last;

  // basic test
  initialize_keys(keys);  initialize_values(values);

  Vector output_keys(keys.size());
  Vector output_values(values.size());

  unique_by_key_copy_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0], 0);
  ASSERT_EQUAL(output_values[1], 2);
  ASSERT_EQUAL(output_values[2], 3);
  ASSERT_EQUAL(output_values[3], 4);
  ASSERT_EQUAL(output_values[4], 7);

  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  unique_by_key_copy_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_unique<T>(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 37);
  
  ASSERT_EQUAL(output_values[0], 0);
  ASSERT_EQUAL(output_values[1], 2);
  ASSERT_EQUAL(output_values[2], 7);
}


void TestUniqueCopyByKeyDeviceSeq()
{
  TestUniqueCopyByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyDeviceSeq);


void TestUniqueCopyByKeyDeviceDevice()
{
  TestUniqueCopyByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyDeviceDevice);


void TestUniqueCopyByKeyDeviceNoSync()
{
  TestUniqueCopyByKeyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyDeviceNoSync);


template<typename ExecutionPolicy>
void TestUniqueCopyByKeyCudaStreams(ExecutionPolicy policy)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector keys;
  Vector values;

  typedef thrust::pair<Vector::iterator, Vector::iterator> iter_pair;
  iter_pair new_last;

  // basic test
  initialize_keys(keys);  initialize_values(values);

  Vector output_keys(keys.size());
  Vector output_values(values.size());

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  new_last = thrust::unique_by_key_copy(streampolicy, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0], 0);
  ASSERT_EQUAL(output_values[1], 2);
  ASSERT_EQUAL(output_values[2], 3);
  ASSERT_EQUAL(output_values[3], 4);
  ASSERT_EQUAL(output_values[4], 7);

  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  new_last = thrust::unique_by_key_copy(streampolicy, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_unique<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 37);
  
  ASSERT_EQUAL(output_values[0], 0);
  ASSERT_EQUAL(output_values[1], 2);
  ASSERT_EQUAL(output_values[2], 7);

  cudaStreamDestroy(s);
}

void TestUniqueCopyByKeyCudaStreamsSync()
{
  TestUniqueCopyByKeyCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyCudaStreamsSync);


void TestUniqueCopyByKeyCudaStreamsNoSync()
{
  TestUniqueCopyByKeyCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyCudaStreamsNoSync);

