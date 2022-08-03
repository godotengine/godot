#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include <iterator>
#include <iostream>

// This example demonstrates how to expand an input sequence by 
// replicating each element a variable number of times. For example,
//
//   expand([2,2,2],[A,B,C]) -> [A,A,B,B,C,C]
//   expand([3,0,1],[A,B,C]) -> [A,A,A,C]
//   expand([1,3,2],[A,B,C]) -> [A,B,B,B,C,C]
//
// The element counts are assumed to be non-negative integers

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator>
OutputIterator expand(InputIterator1 first1,
                      InputIterator1 last1,
                      InputIterator2 first2,
                      OutputIterator output)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
  
  difference_type input_size  = thrust::distance(first1, last1);
  difference_type output_size = thrust::reduce(first1, last1);

  // scan the counts to obtain output offsets for each input element
  thrust::device_vector<difference_type> output_offsets(input_size, 0);
  thrust::exclusive_scan(first1, last1, output_offsets.begin()); 

  // scatter the nonzero counts into their corresponding output positions
  thrust::device_vector<difference_type> output_indices(output_size, 0);
  thrust::scatter_if
    (thrust::counting_iterator<difference_type>(0),
     thrust::counting_iterator<difference_type>(input_size),
     output_offsets.begin(),
     first1,
     output_indices.begin());

  // compute max-scan over the output indices, filling in the holes
  thrust::inclusive_scan
    (output_indices.begin(),
     output_indices.end(),
     output_indices.begin(),
     thrust::maximum<difference_type>());

  // gather input values according to index array (output = first2[output_indices])
  thrust::gather(output_indices.begin(),
                 output_indices.end(),
                 first2,
                 output);

  // return output + output_size
  thrust::advance(output, output_size);
  return output;
}

template <typename Vector>
void print(const std::string& s, const Vector& v)
{
  typedef typename Vector::value_type T;

  std::cout << s;
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

int main(void)
{
  int counts[] = {3,5,2,0,1,3,4,2,4};
  int values[] = {1,2,3,4,5,6,7,8,9};

  size_t input_size  = sizeof(counts) / sizeof(int);
  size_t output_size = thrust::reduce(counts, counts + input_size);

  // copy inputs to device
  thrust::device_vector<int> d_counts(counts, counts + input_size);
  thrust::device_vector<int> d_values(values, values + input_size);
  thrust::device_vector<int> d_output(output_size);

  // expand values according to counts
  expand(d_counts.begin(), d_counts.end(),
         d_values.begin(),
         d_output.begin());

  std::cout << "Expanding values according to counts" << std::endl;
  print(" counts ", d_counts);
  print(" values ", d_values);
  print(" output ", d_output);

  return 0;
}
