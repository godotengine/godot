#include <thrust/detail/raw_reference_cast.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <iostream>

// This example illustrates how to use the raw_reference_cast to convert
// system-specific reference wrappers into native references.
//
// Using iterators in the manner described here is generally discouraged.
// Users should only resort to this technique if there is no viable
// implemention of a given operation in terms of Thrust algorithms.
// For example this particular example is better solved with thrust::copy,
// which is safer and potentially faster.  Only use this approach after all
// safer alternatives have been exhausted.
//
// When a Thrust iterator is referenced (e.g. *iter) the result is not
// a native or "raw" reference like int& or float&.  Instead,
// the result is a type such as thrust::system::cuda::reference<int>
// or thrust::system::tbb::reference<float>, depending on the system
// to which the data belongs.  These reference wrappers are necessary
// to make expressions like *iter1 = *iter2; work correctly when
// iter1 and iter2 refer to data in different memory spaces on
// heterogenous systems.
//
// The raw_reference_cast function essentially strips away the system-specific
// meta-data so it should only be used when the code is guaranteed to be
// executed within an appropriate context.


__host__ __device__
void assign_reference_to_reference(int& x, int& y)
{
  y = x;
}

__host__ __device__
void assign_value_to_reference(int x, int& y)
{
  y = x;
}

template <typename InputIterator,
          typename OutputIterator>
struct copy_iterators
{
  InputIterator  input;
  OutputIterator output;

  copy_iterators(InputIterator input, OutputIterator output)
    : input(input), output(output)
  {}

  __host__ __device__
  void operator()(int i)
  {
    InputIterator  in  = input  + i;
    OutputIterator out = output + i;

    // invalid - reference<int> is not convertible to int&
    // assign_reference_to_reference(*in, *out);
   
    // valid - reference<int> explicitly converted to int&
    assign_reference_to_reference(thrust::raw_reference_cast(*in), thrust::raw_reference_cast(*out));

    // valid - since reference<int> is convertible to int
    assign_value_to_reference(*in, thrust::raw_reference_cast(*out));
  }
};

template <typename Vector>
void print(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;

  std::cout << name << ": ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));  
  std::cout << "\n";
}

int main(void)
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::iterator           Iterator;
  typedef thrust::device_system_tag  System;

  // allocate device memory
  Vector A(5);
  Vector B(5);

  // initialize A and B
  thrust::sequence(A.begin(), A.end());
  thrust::fill(B.begin(), B.end(), 0);

  std::cout << "Before A->B Copy" << std::endl;
  print("A", A);
  print("B", B);

  // note: we must specify the System to ensure correct execution
  thrust::for_each(thrust::counting_iterator<int,System>(0),
                   thrust::counting_iterator<int,System>(5),
                   copy_iterators<Iterator,Iterator>(A.begin(), B.begin()));
  
  std::cout << "After A->B Copy" << std::endl;
  print("A", A);
  print("B", B);
 
  return 0;
}

