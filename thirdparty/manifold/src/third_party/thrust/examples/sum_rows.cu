#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <iostream>

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
  T C; // number of columns
  
  __host__ __device__
  linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i)
  {
    return i / C;
  }
};

int main(void)
{
  int R = 5;     // number of rows
  int C = 8;     // number of columns
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(10, 99);

  // initialize data
  thrust::device_vector<int> array(R * C);
  for (size_t i = 0; i < array.size(); i++)
    array[i] = dist(rng);
  
  // allocate storage for row sums and indices
  thrust::device_vector<int> row_sums(R);
  thrust::device_vector<int> row_indices(R);
  
  // compute row sums by summing values with equal row indices
  thrust::reduce_by_key
    (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)),
     thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)) + (R*C),
     array.begin(),
     row_indices.begin(),
     row_sums.begin(),
     thrust::equal_to<int>(),
     thrust::plus<int>());

  // print data 
  for(int i = 0; i < R; i++)
  {
    std::cout << "[ ";
    for(int j = 0; j < C; j++)
      std::cout << array[i * C + j] << " ";
    std::cout << "] = " << row_sums[i] << "\n";
  }

  return 0;
}

