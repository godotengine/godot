#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <unittest/unittest.h>

#include <cstdint>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5>
__global__
void reduce_by_key_kernel(ExecutionPolicy exec,
                          Iterator1 keys_first, Iterator1 keys_last,
                          Iterator2 values_first,
                          Iterator3 keys_result,
                          Iterator4 values_result,
                          Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename BinaryPredicate, typename Iterator5>
__global__
void reduce_by_key_kernel(ExecutionPolicy exec,
                          Iterator1 keys_first, Iterator1 keys_last,
                          Iterator2 values_first,
                          Iterator3 keys_result,
                          Iterator4 values_result,
                          BinaryPredicate pred,
                          Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result, pred);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename BinaryPredicate, typename BinaryFunction, typename Iterator5>
__global__
void reduce_by_key_kernel(ExecutionPolicy exec,
                          Iterator1 keys_first, Iterator1 keys_last,
                          Iterator2 values_first,
                          Iterator3 keys_result,
                          Iterator4 values_result,
                          BinaryPredicate pred,
                          BinaryFunction binary_op,
                          Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result, pred, binary_op);
}


template<typename T>
struct is_equal_div_10_reduce
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


template<typename ExecutionPolicy>
void TestReduceByKeyDevice(ExecutionPolicy exec)
{
  typedef int T;
  
  thrust::device_vector<T> keys;
  thrust::device_vector<T> values;

  typedef typename thrust::pair<
    typename thrust::device_vector<T>::iterator,
    typename thrust::device_vector<T>::iterator
  > iterator_pair;

  thrust::device_vector<iterator_pair> new_last_vec(1);
  iterator_pair new_last;
  
  // basic test
  initialize_keys(keys);  initialize_values(values);
  
  thrust::device_vector<T> output_keys(keys.size());
  thrust::device_vector<T> output_values(values.size());
  
  reduce_by_key_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), new_last_vec.begin());
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
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);
  
  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  reduce_by_key_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_reduce<T>(), new_last_vec.begin());
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
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1], 20);
  ASSERT_EQUAL(output_values[2], 15);
  
  // test BinaryFunction
  initialize_keys(keys);  initialize_values(values);
  
  reduce_by_key_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), thrust::equal_to<T>(), thrust::plus<T>(), new_last_vec.begin());
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
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);
}


void TestReduceByKeyDeviceSeq()
{
  TestReduceByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReduceByKeyDeviceSeq);


void TestReduceByKeyDeviceDevice()
{
  TestReduceByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestReduceByKeyDeviceDevice);


void TestReduceByKeyDeviceNoSync()
{
  TestReduceByKeyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestReduceByKeyDeviceNoSync);


template<typename ExecutionPolicy>
void TestReduceByKeyCudaStreams(ExecutionPolicy policy)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector keys;
  Vector values;

  thrust::pair<Vector::iterator, Vector::iterator> new_last;

  // basic test
  initialize_keys(keys);  initialize_values(values);

  Vector output_keys(keys.size());
  Vector output_values(values.size());

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  new_last = thrust::reduce_by_key(streampolicy, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);

  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  new_last = thrust::reduce_by_key(streampolicy, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_reduce<T>());

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1], 20);
  ASSERT_EQUAL(output_values[2], 15);

  // test BinaryFunction
  initialize_keys(keys);  initialize_values(values);

  new_last = thrust::reduce_by_key(streampolicy, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), thrust::equal_to<T>(), thrust::plus<T>());

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);

  cudaStreamDestroy(s);
}

void TestReduceByKeyCudaStreamsSync()
{
  TestReduceByKeyCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestReduceByKeyCudaStreamsSync);


void TestReduceByKeyCudaStreamsNoSync()
{
  TestReduceByKeyCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestReduceByKeyCudaStreamsNoSync);


// Maps indices to key ids
class div_op : public thrust::unary_function<std::int64_t, std::int64_t>
{
  std::int64_t m_divisor;

public:
  __host__ div_op(std::int64_t divisor)
    : m_divisor(divisor)
  {}

  __host__ __device__
  std::int64_t operator()(std::int64_t x) const
  {
    return x / m_divisor;
  }
};

// Produces unique sequence for key
class mod_op : public thrust::unary_function<std::int64_t, std::int64_t>
{
  std::int64_t m_divisor;

public:
  __host__ mod_op(std::int64_t divisor)
    : m_divisor(divisor)
  {}

  __host__ __device__
  std::int64_t operator()(std::int64_t x) const
  {
    // div: 2          
    // idx: 0 1   2 3   4 5 
    // key: 0 0 | 1 1 | 2 2 
    // mod: 0 1 | 0 1 | 0 1
    // ret: 0 1   1 2   2 3
    return (x % m_divisor) + (x / m_divisor);
  }
};


void TestReduceByKeyWithBigIndexesHelper(int magnitude)
{
  const std::int64_t key_size_magnitude = 8;
  ASSERT_EQUAL(true, key_size_magnitude < magnitude);

  const std::int64_t num_items       = 1ll << magnitude;
  const std::int64_t num_unique_keys = 1ll << key_size_magnitude;

  // Size of each key group
  const std::int64_t key_size = num_items / num_unique_keys;

  using counting_it      = thrust::counting_iterator<std::int64_t>;
  using transform_key_it = thrust::transform_iterator<div_op, counting_it>;
  using transform_val_it = thrust::transform_iterator<mod_op, counting_it>;

  counting_it count_begin(0ll);
  counting_it count_end = count_begin + num_items;
  ASSERT_EQUAL(static_cast<std::int64_t>(thrust::distance(count_begin, count_end)),
               num_items);

  transform_key_it keys_begin(count_begin, div_op{key_size});
  transform_key_it keys_end(count_end, div_op{key_size});

  transform_val_it values_begin(count_begin, mod_op{key_size});

  thrust::device_vector<std::int64_t> output_keys(num_unique_keys);
  thrust::device_vector<std::int64_t> output_values(num_unique_keys);

  // example:
  //  items:        6
  //  unique_keys:  2
  //  key_size:     3
  //  keys:         0 0 0 | 1 1 1 
  //  values:       0 1 2 | 1 2 3
  //  result:       3       6     = sum(range(key_size)) + key_size * key_id
  thrust::reduce_by_key(keys_begin,
                        keys_end,
                        values_begin,
                        output_keys.begin(),
                        output_values.begin());

  ASSERT_EQUAL(
    true,
    thrust::equal(output_keys.begin(), output_keys.end(), count_begin));

  thrust::host_vector<std::int64_t> result = output_values;

  const std::int64_t sum = (key_size - 1) * key_size / 2;
  for (std::int64_t key_id = 0; key_id < num_unique_keys; key_id++)
  {
    ASSERT_EQUAL(result[key_id], sum + key_id * key_size);
  }
}

void TestReduceByKeyWithBigIndexes()
{
  TestReduceByKeyWithBigIndexesHelper(30);
  TestReduceByKeyWithBigIndexesHelper(31);
  TestReduceByKeyWithBigIndexesHelper(32);
  TestReduceByKeyWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestReduceByKeyWithBigIndexes);
