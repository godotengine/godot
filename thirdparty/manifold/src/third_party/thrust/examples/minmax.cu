#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/random.h>


// compute minimum and maximum values in a single reduction

// minmax_pair stores the minimum and maximum 
// values that have been encountered so far
template <typename T>
struct minmax_pair
{
  T min_val;
  T max_val;
};

// minmax_unary_op is a functor that takes in a value x and
// returns a minmax_pair whose minimum and maximum values
// are initialized to x.
template <typename T>
struct minmax_unary_op
  : public thrust::unary_function< T, minmax_pair<T> >
{
  __host__ __device__
  minmax_pair<T> operator()(const T& x) const
  {
    minmax_pair<T> result;
    result.min_val = x;
    result.max_val = x;
    return result;
  }
};

// minmax_binary_op is a functor that accepts two minmax_pair 
// structs and returns a new minmax_pair whose minimum and 
// maximum values are the min() and max() respectively of 
// the minimums and maximums of the input pairs
template <typename T>
struct minmax_binary_op
  : public thrust::binary_function< minmax_pair<T>, minmax_pair<T>, minmax_pair<T> >
{
  __host__ __device__
  minmax_pair<T> operator()(const minmax_pair<T>& x, const minmax_pair<T>& y) const
  {
    minmax_pair<T> result;
    result.min_val = thrust::min(x.min_val, y.min_val);
    result.max_val = thrust::max(x.max_val, y.max_val);
    return result;
  }
};


int main(void)
{
  // input size
  size_t N = 10;

  // initialize random number generator
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(10, 99);

  // initialize data on host
  thrust::device_vector<int> data(N);
  for (size_t i = 0; i < data.size(); i++)
      data[i] = dist(rng);

  // setup arguments
  minmax_unary_op<int>  unary_op;
  minmax_binary_op<int> binary_op;

  // initialize reduction with the first value
  minmax_pair<int> init = unary_op(data[0]);

  // compute minimum and maximum values
  minmax_pair<int> result = thrust::transform_reduce(data.begin(), data.end(), unary_op, init, binary_op);

  // print results
  std::cout << "[ ";
  for(size_t i = 0; i < N; i++)
    std::cout << data[i] << " ";
  std::cout << "]" << std::endl;
 
  std::cout << "minimum = " << result.min_val << std::endl;
  std::cout << "maximum = " << result.max_val << std::endl;

  return 0;
}

