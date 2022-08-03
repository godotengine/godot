#include <unittest/unittest.h>
#include <thrust/binary_search.h>

#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>


//////////////////////
// Vector Functions //
//////////////////////

// convert xxx_vector<T1> to xxx_vector<T2> 
template <class ExampleVector, typename NewType> 
struct vector_like
{
    typedef typename ExampleVector::allocator_type alloc;
    typedef typename thrust::detail::allocator_traits<alloc> alloc_traits;
    typedef typename alloc_traits::template rebind_alloc<NewType> new_alloc;
    typedef thrust::detail::vector_base<NewType, new_alloc> type;
};

template <class Vector>
void TestVectorLowerBoundSimple(void)
{
    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    Vector input(10);
    thrust::sequence(input.begin(), input.end());

    typedef typename Vector::difference_type int_type;
    typedef typename vector_like<Vector, int_type>::type IntVector;

    // test with integral output type
    IntVector integral_output(10);
    thrust::lower_bound(vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());
    
    typename IntVector::iterator output_end = thrust::lower_bound(vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    ASSERT_EQUAL((output_end - integral_output.begin()), 10);

    ASSERT_EQUAL(integral_output[0], 0);
    ASSERT_EQUAL(integral_output[1], 1);
    ASSERT_EQUAL(integral_output[2], 1);
    ASSERT_EQUAL(integral_output[3], 2);
    ASSERT_EQUAL(integral_output[4], 2);
    ASSERT_EQUAL(integral_output[5], 2);
    ASSERT_EQUAL(integral_output[6], 3);
    ASSERT_EQUAL(integral_output[7], 3);
    ASSERT_EQUAL(integral_output[8], 4);
    ASSERT_EQUAL(integral_output[9], 5);

//    // test with interator output type
//    typedef typename vector_like<Vector, typename Vector::iterator>::type IteratorVector;
//    IteratorVector iterator_output(10);
//    thrust::lower_bound(vec.begin(), vec.end(), input.begin(), input.end(), iterator_output.begin());
//
//    ASSERT_EQUAL(iterator_output[0] - vec.begin(), 0);
//    ASSERT_EQUAL(iterator_output[1] - vec.begin(), 1);
//    ASSERT_EQUAL(iterator_output[2] - vec.begin(), 1);
//    ASSERT_EQUAL(iterator_output[3] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[4] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[5] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[6] - vec.begin(), 3);
//    ASSERT_EQUAL(iterator_output[7] - vec.begin(), 3);
//    ASSERT_EQUAL(iterator_output[8] - vec.begin(), 4);
//    ASSERT_EQUAL(iterator_output[9] - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestVectorLowerBoundSimple);


template<typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(my_system &system, ForwardIterator,ForwardIterator,InputIterator,InputIterator,OutputIterator output)
{
    system.validate_dispatch();
    return output;
}

void TestVectorLowerBoundDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::lower_bound(sys,
                        vec.begin(),
                        vec.end(),
                        vec.begin(),
                        vec.end(),
                        vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestVectorLowerBoundDispatchExplicit);


template<typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(my_tag, ForwardIterator,ForwardIterator,InputIterator,InputIterator,OutputIterator output)
{
    *output = 13;
    return output;
}

void TestVectorLowerBoundDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::lower_bound(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestVectorLowerBoundDispatchImplicit);


template <class Vector>
void TestVectorUpperBoundSimple(void)
{
    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    Vector input(10);
    thrust::sequence(input.begin(), input.end());

    typedef typename Vector::difference_type int_type;
    typedef typename vector_like<Vector, int_type>::type IntVector;

    // test with integral output type
    IntVector integral_output(10);
    typename IntVector::iterator output_end = thrust::upper_bound(vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    ASSERT_EQUAL((output_end - integral_output.begin()), 10);

    ASSERT_EQUAL(integral_output[0], 1);
    ASSERT_EQUAL(integral_output[1], 1);
    ASSERT_EQUAL(integral_output[2], 2);
    ASSERT_EQUAL(integral_output[3], 2);
    ASSERT_EQUAL(integral_output[4], 2);
    ASSERT_EQUAL(integral_output[5], 3);
    ASSERT_EQUAL(integral_output[6], 3);
    ASSERT_EQUAL(integral_output[7], 4);
    ASSERT_EQUAL(integral_output[8], 5);
    ASSERT_EQUAL(integral_output[9], 5);

//    // test with interator output type
//    typedef typename vector_like<Vector, typename Vector::iterator>::type IteratorVector;
//    IteratorVector iterator_output(10);
//    thrust::lower_bound(vec.begin(), vec.end(), input.begin(), input.end(), iterator_output.begin());
//
//    ASSERT_EQUAL(iterator_output[0] - vec.begin(), 1);
//    ASSERT_EQUAL(iterator_output[1] - vec.begin(), 1);
//    ASSERT_EQUAL(iterator_output[2] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[3] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[4] - vec.begin(), 2);
//    ASSERT_EQUAL(iterator_output[5] - vec.begin(), 3);
//    ASSERT_EQUAL(iterator_output[6] - vec.begin(), 3);
//    ASSERT_EQUAL(iterator_output[7] - vec.begin(), 4);
//    ASSERT_EQUAL(iterator_output[8] - vec.begin(), 5);
//    ASSERT_EQUAL(iterator_output[9] - vec.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestVectorUpperBoundSimple);


template<typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator upper_bound(my_system &system, ForwardIterator,ForwardIterator,InputIterator,InputIterator,OutputIterator output)
{
    system.validate_dispatch();
    return output;
}

void TestVectorUpperBoundDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::upper_bound(sys,
                        vec.begin(),
                        vec.end(),
                        vec.begin(),
                        vec.end(),
                        vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestVectorUpperBoundDispatchExplicit);


template<typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator upper_bound(my_tag, ForwardIterator,ForwardIterator,InputIterator,InputIterator,OutputIterator output)
{
    *output = 13;
    return output;
}

void TestVectorUpperBoundDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::upper_bound(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestVectorUpperBoundDispatchImplicit);


template <class Vector>
void TestVectorBinarySearchSimple(void)
{
    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    Vector input(10);
    thrust::sequence(input.begin(), input.end());

    typedef typename vector_like<Vector, bool>::type BoolVector;
    typedef typename Vector::difference_type int_type;
    typedef typename vector_like<Vector,  int_type>::type IntVector;

    // test with boolean output type
    BoolVector bool_output(10);
    typename BoolVector::iterator bool_output_end = thrust::binary_search(vec.begin(), vec.end(), input.begin(), input.end(), bool_output.begin());

    ASSERT_EQUAL((bool_output_end - bool_output.begin()), 10);

    ASSERT_EQUAL(bool_output[0],  true);
    ASSERT_EQUAL(bool_output[1], false);
    ASSERT_EQUAL(bool_output[2],  true);
    ASSERT_EQUAL(bool_output[3], false);
    ASSERT_EQUAL(bool_output[4], false);
    ASSERT_EQUAL(bool_output[5],  true);
    ASSERT_EQUAL(bool_output[6], false);
    ASSERT_EQUAL(bool_output[7],  true);
    ASSERT_EQUAL(bool_output[8],  true);
    ASSERT_EQUAL(bool_output[9], false);
    
    // test with integral output type
    IntVector integral_output(10, 2);
    typename IntVector::iterator int_output_end = thrust::binary_search(vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    ASSERT_EQUAL((int_output_end - integral_output.begin()), 10);
    
    ASSERT_EQUAL(integral_output[0], 1);
    ASSERT_EQUAL(integral_output[1], 0);
    ASSERT_EQUAL(integral_output[2], 1);
    ASSERT_EQUAL(integral_output[3], 0);
    ASSERT_EQUAL(integral_output[4], 0);
    ASSERT_EQUAL(integral_output[5], 1);
    ASSERT_EQUAL(integral_output[6], 0);
    ASSERT_EQUAL(integral_output[7], 1);
    ASSERT_EQUAL(integral_output[8], 1);
    ASSERT_EQUAL(integral_output[9], 0);
}
DECLARE_VECTOR_UNITTEST(TestVectorBinarySearchSimple);


template<typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator binary_search(my_system &system, ForwardIterator,ForwardIterator,InputIterator,InputIterator,OutputIterator output)
{
    system.validate_dispatch();
    return output;
}

void TestVectorBinarySearchDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::binary_search(sys,
                          vec.begin(),
                          vec.end(),
                          vec.begin(),
                          vec.end(),
                          vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestVectorBinarySearchDispatchExplicit);


template<typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator binary_search(my_tag, ForwardIterator,ForwardIterator,InputIterator,InputIterator,OutputIterator output)
{
    *output = 13;
    return output;
}

void TestVectorBinarySearchDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::binary_search(thrust::retag<my_tag>(vec.begin()),
                          thrust::retag<my_tag>(vec.end()),
                          thrust::retag<my_tag>(vec.begin()),
                          thrust::retag<my_tag>(vec.end()),
                          thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestVectorBinarySearchDispatchImplicit);


template <typename T>
struct TestVectorLowerBound
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_vec = unittest::random_integers<T>(n); thrust::sort(h_vec.begin(), h_vec.end());
    thrust::device_vector<T> d_vec = h_vec;

    thrust::host_vector<T>   h_input = unittest::random_integers<T>(2*n);
    thrust::device_vector<T> d_input = h_input;
    
    typedef typename thrust::host_vector<T>::difference_type int_type;
    thrust::host_vector<int_type>   h_output(2*n);
    thrust::device_vector<int_type> d_output(2*n);

    thrust::lower_bound(h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin());
    thrust::lower_bound(d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestVectorLowerBound, SignedIntegralTypes> TestVectorLowerBoundInstance;


template <typename T>
struct TestVectorUpperBound
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_vec = unittest::random_integers<T>(n); thrust::sort(h_vec.begin(), h_vec.end());
    thrust::device_vector<T> d_vec = h_vec;

    thrust::host_vector<T>   h_input = unittest::random_integers<T>(2*n);
    thrust::device_vector<T> d_input = h_input;
    
    typedef typename thrust::host_vector<T>::difference_type int_type;
    thrust::host_vector<int_type>   h_output(2*n);
    thrust::device_vector<int_type> d_output(2*n);

    thrust::upper_bound(h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin());
    thrust::upper_bound(d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestVectorUpperBound, SignedIntegralTypes> TestVectorUpperBoundInstance;

template <typename T>
struct TestVectorBinarySearch
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_vec = unittest::random_integers<T>(n); thrust::sort(h_vec.begin(), h_vec.end());
    thrust::device_vector<T> d_vec = h_vec;

    thrust::host_vector<T>   h_input = unittest::random_integers<T>(2*n);
    thrust::device_vector<T> d_input = h_input;
    
    typedef typename thrust::host_vector<T>::difference_type int_type;
    thrust::host_vector<int_type>   h_output(2*n);
    thrust::device_vector<int_type> d_output(2*n);

    thrust::binary_search(h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin());
    thrust::binary_search(d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQUAL(h_output, d_output);
  }
};
VariableUnitTest<TestVectorBinarySearch, SignedIntegralTypes> TestVectorBinarySearchInstance;

template <typename T>
struct TestVectorLowerBoundDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_vec = unittest::random_integers<T>(n); thrust::sort(h_vec.begin(), h_vec.end());
    thrust::device_vector<T> d_vec = h_vec;

    thrust::host_vector<T>   h_input = unittest::random_integers<T>(2*n);
    thrust::device_vector<T> d_input = h_input;
    
    thrust::discard_iterator<> h_result =
      thrust::lower_bound(h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), thrust::make_discard_iterator());
    thrust::discard_iterator<> d_result =
      thrust::lower_bound(d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(2*n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestVectorLowerBoundDiscardIterator, SignedIntegralTypes> TestVectorLowerBoundDiscardIteratorInstance;


template <typename T>
struct TestVectorUpperBoundDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_vec = unittest::random_integers<T>(n); thrust::sort(h_vec.begin(), h_vec.end());
    thrust::device_vector<T> d_vec = h_vec;

    thrust::host_vector<T>   h_input = unittest::random_integers<T>(2*n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> h_result =
      thrust::upper_bound(h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), thrust::make_discard_iterator());
    thrust::discard_iterator<> d_result =
      thrust::upper_bound(d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(2*n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestVectorUpperBoundDiscardIterator, SignedIntegralTypes> TestVectorUpperBoundDiscardIteratorInstance;

template <typename T>
struct TestVectorBinarySearchDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T>   h_vec = unittest::random_integers<T>(n); thrust::sort(h_vec.begin(), h_vec.end());
    thrust::device_vector<T> d_vec = h_vec;

    thrust::host_vector<T>   h_input = unittest::random_integers<T>(2*n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> h_result =
      thrust::binary_search(h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), thrust::make_discard_iterator());
    thrust::discard_iterator<> d_result =
      thrust::binary_search(d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(2*n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestVectorBinarySearchDiscardIterator, SignedIntegralTypes> TestVectorBinarySearchDiscardIteratorInstance;

