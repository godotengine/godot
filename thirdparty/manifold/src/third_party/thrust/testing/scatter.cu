#include <unittest/unittest.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <algorithm>

template <class Vector>
void TestScatterSimple(void)
{
    Vector map(5);  // scatter indices
    Vector src(5);  // source vector
    Vector dst(8);  // destination vector

    map[0] = 6; map[1] = 3; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;

    thrust::scatter(src.begin(), src.end(), map.begin(), dst.begin());

    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 4);
    ASSERT_EQUAL(dst[3], 1);
    ASSERT_EQUAL(dst[4], 0);
    ASSERT_EQUAL(dst[5], 0);
    ASSERT_EQUAL(dst[6], 0);
    ASSERT_EQUAL(dst[7], 3);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestScatterSimple);


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
void scatter(my_system &system,
             InputIterator1,
             InputIterator1,
             InputIterator2,
             RandomAccessIterator)
{
    system.validate_dispatch();
}


void TestScatterDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::scatter(sys,
                    vec.begin(),
                    vec.begin(),
                    vec.begin(),
                    vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestScatterDispatchExplicit);


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
void scatter(my_tag,
             InputIterator1,
             InputIterator1,
             InputIterator2,
             RandomAccessIterator output)
{
    *output = 13;
}

void TestScatterDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::scatter(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScatterDispatchImplicit);


template <typename T>
void TestScatter(const size_t n)
{
    const size_t output_size = std::min((size_t) 10, 2 * n);
    
    thrust::host_vector<T> h_input(n, (T) 1);
    thrust::device_vector<T> d_input(n, (T) 1);
   
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % output_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    thrust::host_vector<T>   h_output(output_size, (T) 0);
    thrust::device_vector<T> d_output(output_size, (T) 0);

    thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());
    thrust::scatter(d_input.begin(), d_input.end(), d_map.begin(), d_output.begin());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestScatter);


template <typename T>
void TestScatterToDiscardIterator(const size_t n)
{
    const size_t output_size = std::min((size_t) 10, 2 * n);
    
    thrust::host_vector<T> h_input(n, (T) 1);
    thrust::device_vector<T> d_input(n, (T) 1);
   
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % output_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), thrust::make_discard_iterator());
    thrust::scatter(d_input.begin(), d_input.end(), d_map.begin(), thrust::make_discard_iterator());

    // there's nothing to check -- just make sure it compiles
}
DECLARE_VARIABLE_UNITTEST(TestScatterToDiscardIterator);


template <class Vector>
void TestScatterIfSimple(void)
{
    Vector flg(5);  // predicate array
    Vector map(5);  // scatter indices
    Vector src(5);  // source vector
    Vector dst(8);  // destination vector

    flg[0] = 0; flg[1] = 1; flg[2] = 0; flg[3] = 1; flg[4] = 0;
    map[0] = 6; map[1] = 3; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;

    thrust::scatter_if(src.begin(), src.end(), map.begin(), flg.begin(), dst.begin());

    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 0);
    ASSERT_EQUAL(dst[2], 0);
    ASSERT_EQUAL(dst[3], 1);
    ASSERT_EQUAL(dst[4], 0);
    ASSERT_EQUAL(dst[5], 0);
    ASSERT_EQUAL(dst[6], 0);
    ASSERT_EQUAL(dst[7], 3);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestScatterIfSimple);


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
void scatter_if(my_system &system,
                InputIterator1,
                InputIterator1,
                InputIterator2,
                InputIterator3,
                RandomAccessIterator)
{
    system.validate_dispatch();
}

void TestScatterIfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::scatter_if(sys,
                       vec.begin(),
                       vec.begin(),
                       vec.begin(),
                       vec.begin(),
                       vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestScatterIfDispatchExplicit);


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
void scatter_if(my_tag,
                InputIterator1,
                InputIterator1,
                InputIterator2,
                InputIterator3,
                RandomAccessIterator output)
{
    *output = 13;
}

void TestScatterIfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::scatter_if(thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestScatterIfDispatchImplicit);


template <typename T>
class is_even_scatter_if
{
    public:
    __host__ __device__ bool operator()(const T i) const { return (i % 2) == 0; }
};

template <typename T>
void TestScatterIf(const size_t n)
{
    const size_t output_size = std::min((size_t) 10, 2 * n);
    
    thrust::host_vector<T> h_input(n, (T) 1);
    thrust::device_vector<T> d_input(n, (T) 1);
   
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % output_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    thrust::host_vector<T>   h_output(output_size, (T) 0);
    thrust::device_vector<T> d_output(output_size, (T) 0);

    thrust::scatter_if(h_input.begin(), h_input.end(), h_map.begin(), h_map.begin(), h_output.begin(), is_even_scatter_if<unsigned int>());
    thrust::scatter_if(d_input.begin(), d_input.end(), d_map.begin(), d_map.begin(), d_output.begin(), is_even_scatter_if<unsigned int>());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestScatterIf);


template <typename T>
void TestScatterIfToDiscardIterator(const size_t n)
{
    const size_t output_size = std::min((size_t) 10, 2 * n);
    
    thrust::host_vector<T> h_input(n, (T) 1);
    thrust::device_vector<T> d_input(n, (T) 1);
   
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % output_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    thrust::scatter_if(h_input.begin(), h_input.end(), h_map.begin(), h_map.begin(), thrust::make_discard_iterator(), is_even_scatter_if<unsigned int>());
    thrust::scatter_if(d_input.begin(), d_input.end(), d_map.begin(), d_map.begin(), thrust::make_discard_iterator(), is_even_scatter_if<unsigned int>());
}
DECLARE_VARIABLE_UNITTEST(TestScatterIfToDiscardIterator);


template <typename Vector>
void TestScatterCountingIterator(void)
{
    Vector source(10);
    thrust::sequence(source.begin(), source.end(), 0);

    Vector map(10);
    thrust::sequence(map.begin(), map.end(), 0);

    Vector output(10);

    // source has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10),
                    map.begin(),
                    output.begin());

    ASSERT_EQUAL(output, map);
    
    // map has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter(source.begin(), source.end(),
                    thrust::make_counting_iterator(0),
                    output.begin());

    ASSERT_EQUAL(output, map);
    
    // source and map have any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10),
                    thrust::make_counting_iterator(0),
                    output.begin());

    ASSERT_EQUAL(output, map);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestScatterCountingIterator);


template <typename Vector>
void TestScatterIfCountingIterator(void)
{
    Vector source(10);
    thrust::sequence(source.begin(), source.end(), 0);

    Vector map(10);
    thrust::sequence(map.begin(), map.end(), 0);
    
    Vector stencil(10, 1);

    Vector output(10);

    // source has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10),
                       map.begin(),
                       stencil.begin(),
                       output.begin());

    ASSERT_EQUAL(output, map);
    
    // map has any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter_if(source.begin(), source.end(),
                       thrust::make_counting_iterator(0),
                       stencil.begin(),
                       output.begin());

    ASSERT_EQUAL(output, map);
    
    // source and map have any_system_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10),
                       thrust::make_counting_iterator(0),
                       stencil.begin(),
                       output.begin());

    ASSERT_EQUAL(output, map);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestScatterIfCountingIterator);

