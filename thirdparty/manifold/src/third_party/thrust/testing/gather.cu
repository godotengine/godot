#include <unittest/unittest.h>
#include <thrust/gather.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>
#include <algorithm>

THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN


template <class Vector>
void TestGatherSimple(void)
{
    Vector map(5);  // gather indices
    Vector src(8);  // source vector
    Vector dst(5);  // destination vector

    map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

    thrust::gather(map.begin(), map.end(), src.begin(), dst.begin());

    ASSERT_EQUAL(dst[0], 6);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 1);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 2);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestGatherSimple);


template<typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
OutputIterator gather(my_system &system, InputIterator, InputIterator, RandomAccessIterator, OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

void TestGatherDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::gather(sys,
                   vec.begin(),
                   vec.end(),
                   vec.begin(),
                   vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestGatherDispatchExplicit);


template<typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
OutputIterator gather(my_tag, InputIterator, InputIterator, RandomAccessIterator, OutputIterator result)
{
    *result = 13;
    return result;
}

void TestGatherDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::gather(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.end()),
                   thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestGatherDispatchImplicit);


template <typename T>
void TestGather(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    // gather destination
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), h_output.begin());
    thrust::gather(d_map.begin(), d_map.end(), d_source.begin(), d_output.begin());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestGather);


template <typename T>
void TestGatherToDiscardIterator(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    thrust::discard_iterator<> h_result = 
      thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), thrust::make_discard_iterator());

    thrust::discard_iterator<> d_result =
      thrust::gather(d_map.begin(), d_map.end(), d_source.begin(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestGatherToDiscardIterator);


template <class Vector>
void TestGatherIfSimple(void)
{
    Vector flg(5);  // predicate array
    Vector map(5);  // gather indices
    Vector src(8);  // source vector
    Vector dst(5);  // destination vector

    flg[0] = 0; flg[1] = 1; flg[2] = 0; flg[3] = 1; flg[4] = 0;
    map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

    thrust::gather_if(map.begin(), map.end(), flg.begin(), src.begin(), dst.begin());

    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 0);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 0);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestGatherIfSimple);

template <typename T>
struct is_even_gather_if
{
    __host__ __device__
    bool operator()(const T i) const
    { 
        return (i % 2) == 0;
    }
};

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
OutputIterator gather_if(my_system &system,
                         InputIterator1, //       map_first,
                         InputIterator1, //       map_last,
                         InputIterator2, //       stencil,
                         RandomAccessIterator, // input_first,
                         OutputIterator       result)
{
    system.validate_dispatch();
    return result;
}

void TestGatherIfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::gather_if(sys,
                      vec.begin(),
                      vec.end(),
                      vec.begin(),
                      vec.begin(),
                      vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestGatherIfDispatchExplicit);


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
OutputIterator gather_if(my_tag,
                         InputIterator1, //       map_first,
                         InputIterator1, //       map_last,
                         InputIterator2, //       stencil,
                         RandomAccessIterator, // input_first,
                         OutputIterator       result)
{
    *result = 13;
    return result;
}

void TestGatherIfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::gather_if(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.end()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestGatherIfDispatchImplicit);


template <typename T>
void TestGatherIf(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] = h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;
    
    // gather stencil
    thrust::host_vector<unsigned int> h_stencil = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_stencil[i] = h_stencil[i] % 2;
    
    thrust::device_vector<unsigned int> d_stencil = h_stencil;

    // gather destination
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_source.begin(), h_output.begin(), is_even_gather_if<unsigned int>());
    thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(), d_source.begin(), d_output.begin(), is_even_gather_if<unsigned int>());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestGatherIf);



template <typename T>
void TestGatherIfToDiscardIterator(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] = h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;
    
    // gather stencil
    thrust::host_vector<unsigned int> h_stencil = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_stencil[i] = h_stencil[i] % 2;
    
    thrust::device_vector<unsigned int> d_stencil = h_stencil;

    thrust::discard_iterator<> h_result =
      thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_source.begin(), thrust::make_discard_iterator(), is_even_gather_if<unsigned int>());

    thrust::discard_iterator<> d_result =
      thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(), d_source.begin(), thrust::make_discard_iterator(), is_even_gather_if<unsigned int>());

    thrust::discard_iterator<> reference(n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestGatherIfToDiscardIterator);


template <typename Vector>
void TestGatherCountingIterator(void)
{
    Vector source(10);
    thrust::sequence(source.begin(), source.end(), 0);

    Vector map(10);
    thrust::sequence(map.begin(), map.end(), 0);

    Vector output(10);

    // source has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::gather(map.begin(),
                   map.end(),
                   thrust::make_counting_iterator(0),
                   output.begin());

    ASSERT_EQUAL(output, map);
    
    // map has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::gather(thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator((int)source.size()),
                   source.begin(),
                   output.begin());

    ASSERT_EQUAL(output, map);
    
    // source and map have any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::gather(thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator((int)output.size()),
                   thrust::make_counting_iterator(0),
                   output.begin());

    ASSERT_EQUAL(output, map);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestGatherCountingIterator);

THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END
