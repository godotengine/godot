#include <unittest/unittest.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>

template <class Vector>
void TestPermutationIteratorSimple(void)
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator   Iterator;

    Vector source(8);
    Vector indices(4);
    
    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;
   
    thrust::permutation_iterator<Iterator, Iterator> begin(source.begin(), indices.begin());
    thrust::permutation_iterator<Iterator, Iterator> end(source.begin(),   indices.end());

    ASSERT_EQUAL(end - begin, 4);
    ASSERT_EQUAL((begin + 4) == end, true);

    ASSERT_EQUAL((T) *begin, 4);

    begin++;
    end--;

    ASSERT_EQUAL((T) *begin, 1);
    ASSERT_EQUAL((T) *end,   8);
    ASSERT_EQUAL(end - begin, 2);

    end--;

    *begin = 10;
    *end   = 20;

    ASSERT_EQUAL(source[0], 10);
    ASSERT_EQUAL(source[1],  2);
    ASSERT_EQUAL(source[2],  3);
    ASSERT_EQUAL(source[3],  4);
    ASSERT_EQUAL(source[4],  5);
    ASSERT_EQUAL(source[5], 20);
    ASSERT_EQUAL(source[6],  7);
    ASSERT_EQUAL(source[7],  8);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorSimple);

template <class Vector>
void TestPermutationIteratorGather(void)
{
    typedef typename Vector::iterator Iterator;

    Vector source(8);
    Vector indices(4);
    Vector output(4, 10);
    
    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;
   
    thrust::permutation_iterator<Iterator, Iterator> p_source(source.begin(), indices.begin());

    thrust::copy(p_source, p_source + 4, output.begin());

    ASSERT_EQUAL(output[0], 4);
    ASSERT_EQUAL(output[1], 1);
    ASSERT_EQUAL(output[2], 6);
    ASSERT_EQUAL(output[3], 8);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorGather);

template <class Vector>
void TestPermutationIteratorScatter(void)
{
    typedef typename Vector::iterator Iterator;

    Vector source(4, 10);
    Vector indices(4);
    Vector output(8);
    
    // initialize output
    thrust::sequence(output.begin(), output.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;
   
    // construct transform_iterator
    thrust::permutation_iterator<Iterator, Iterator> p_output(output.begin(), indices.begin());

    thrust::copy(source.begin(), source.end(), p_output);

    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1],  2);
    ASSERT_EQUAL(output[2],  3);
    ASSERT_EQUAL(output[3], 10);
    ASSERT_EQUAL(output[4],  5);
    ASSERT_EQUAL(output[5], 10);
    ASSERT_EQUAL(output[6],  7);
    ASSERT_EQUAL(output[7], 10);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorScatter);

template <class Vector>
void TestMakePermutationIterator(void)
{
    Vector source(8);
    Vector indices(4);
    Vector output(4, 10);
    
    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;
   
    thrust::copy(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                 thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
                 output.begin());

    ASSERT_EQUAL(output[0], 4);
    ASSERT_EQUAL(output[1], 1);
    ASSERT_EQUAL(output[2], 6);
    ASSERT_EQUAL(output[3], 8);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestMakePermutationIterator);

template <typename Vector>
void TestPermutationIteratorReduce(void)
{
    typedef typename Vector::value_type T;
    typedef typename Vector::iterator Iterator;

    Vector source(8);
    Vector indices(4);
    Vector output(4, 10);
    
    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;
   
    // construct transform_iterator
    thrust::permutation_iterator<Iterator, Iterator> iter(source.begin(), indices.begin());

    T result1 = thrust::reduce(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                               thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4);

    ASSERT_EQUAL(result1, 19);
    
    T result2 = thrust::transform_reduce(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                                         thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
                                         thrust::negate<T>(),
                                         T(0),
                                         thrust::plus<T>());
    ASSERT_EQUAL(result2, -19);
};
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorReduce);

void TestPermutationIteratorHostDeviceGather(void)
{
    typedef int T;
    typedef thrust::host_vector<T> HostVector;
    typedef thrust::host_vector<T> DeviceVector;
    typedef HostVector::iterator   HostIterator;
    typedef DeviceVector::iterator DeviceIterator;

    HostVector h_source(8);
    HostVector h_indices(4);
    HostVector h_output(4, 10);
    
    DeviceVector d_source(8);
    DeviceVector d_indices(4);
    DeviceVector d_output(4, 10);

    // initialize source
    thrust::sequence(h_source.begin(), h_source.end(), 1);
    thrust::sequence(d_source.begin(), d_source.end(), 1);

    h_indices[0] = d_indices[0] = 3;
    h_indices[1] = d_indices[1] = 0;
    h_indices[2] = d_indices[2] = 5;
    h_indices[3] = d_indices[3] = 7;
   
    thrust::permutation_iterator<HostIterator,   HostIterator>   p_h_source(h_source.begin(), h_indices.begin());
    thrust::permutation_iterator<DeviceIterator, DeviceIterator> p_d_source(d_source.begin(), d_indices.begin());

    // gather host->device
    thrust::copy(p_h_source, p_h_source + 4, d_output.begin());

    ASSERT_EQUAL(d_output[0], 4);
    ASSERT_EQUAL(d_output[1], 1);
    ASSERT_EQUAL(d_output[2], 6);
    ASSERT_EQUAL(d_output[3], 8);
    
    // gather device->host
    thrust::copy(p_d_source, p_d_source + 4, h_output.begin());

    ASSERT_EQUAL(h_output[0], 4);
    ASSERT_EQUAL(h_output[1], 1);
    ASSERT_EQUAL(h_output[2], 6);
    ASSERT_EQUAL(h_output[3], 8);
}
DECLARE_UNITTEST(TestPermutationIteratorHostDeviceGather);

void TestPermutationIteratorHostDeviceScatter(void)
{
    typedef int T;
    typedef thrust::host_vector<T> HostVector;
    typedef thrust::host_vector<T> DeviceVector;
    typedef HostVector::iterator   HostIterator;
    typedef DeviceVector::iterator DeviceIterator;

    HostVector h_source(4,10);
    HostVector h_indices(4);
    HostVector h_output(8);
    
    DeviceVector d_source(4,10);
    DeviceVector d_indices(4);
    DeviceVector d_output(8);

    // initialize source
    thrust::sequence(h_output.begin(), h_output.end(), 1);
    thrust::sequence(d_output.begin(), d_output.end(), 1);

    h_indices[0] = d_indices[0] = 3;
    h_indices[1] = d_indices[1] = 0;
    h_indices[2] = d_indices[2] = 5;
    h_indices[3] = d_indices[3] = 7;
   
    thrust::permutation_iterator<HostIterator,   HostIterator>   p_h_output(h_output.begin(), h_indices.begin());
    thrust::permutation_iterator<DeviceIterator, DeviceIterator> p_d_output(d_output.begin(), d_indices.begin());

    // scatter host->device
    thrust::copy(h_source.begin(), h_source.end(), p_d_output);

    ASSERT_EQUAL(d_output[0], 10);
    ASSERT_EQUAL(d_output[1],  2);
    ASSERT_EQUAL(d_output[2],  3);
    ASSERT_EQUAL(d_output[3], 10);
    ASSERT_EQUAL(d_output[4],  5);
    ASSERT_EQUAL(d_output[5], 10);
    ASSERT_EQUAL(d_output[6],  7);
    ASSERT_EQUAL(d_output[7], 10);
    
    // scatter device->host
    thrust::copy(d_source.begin(), d_source.end(), p_h_output);

    ASSERT_EQUAL(h_output[0], 10);
    ASSERT_EQUAL(h_output[1],  2);
    ASSERT_EQUAL(h_output[2],  3);
    ASSERT_EQUAL(h_output[3], 10);
    ASSERT_EQUAL(h_output[4],  5);
    ASSERT_EQUAL(h_output[5], 10);
    ASSERT_EQUAL(h_output[6],  7);
    ASSERT_EQUAL(h_output[7], 10);
}
DECLARE_UNITTEST(TestPermutationIteratorHostDeviceScatter);

template <typename Vector>
void TestPermutationIteratorWithCountingIterator(void)
{
  using T = typename Vector::value_type;
  using diff_t = typename thrust::counting_iterator<T>::difference_type;
  
  thrust::counting_iterator<T> input(0), index(0);

  // test copy()
  {
    Vector output(4,0);

    auto first = thrust::make_permutation_iterator(input, index);
    auto last  = thrust::make_permutation_iterator(input,
                                                   index + static_cast<diff_t>(output.size()));

    thrust::copy(first, last, output.begin());

    ASSERT_EQUAL(output[0], 0);
    ASSERT_EQUAL(output[1], 1);
    ASSERT_EQUAL(output[2], 2);
    ASSERT_EQUAL(output[3], 3);
  }

  // test copy()
  {
    Vector output(4,0);

    thrust::transform(thrust::make_permutation_iterator(input, index),
                      thrust::make_permutation_iterator(input, index + 4),
                      output.begin(),
                      thrust::identity<T>());

    ASSERT_EQUAL(output[0], 0);
    ASSERT_EQUAL(output[1], 1);
    ASSERT_EQUAL(output[2], 2);
    ASSERT_EQUAL(output[3], 3);
  }
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestPermutationIteratorWithCountingIterator);

