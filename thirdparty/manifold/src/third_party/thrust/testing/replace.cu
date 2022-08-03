#include <unittest/unittest.h>
#include <thrust/replace.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>


template <class Vector>
void TestReplaceSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    thrust::replace(data.begin(), data.end(), (T) 1, (T) 4);
    thrust::replace(data.begin(), data.end(), (T) 2, (T) 5);

    Vector result(5);
    result[0] =  4; 
    result[1] =  5; 
    result[2] =  4;
    result[3] =  3; 
    result[4] =  5; 

    ASSERT_EQUAL(data, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceSimple);


template<typename ForwardIterator, typename T>
void replace(my_system &system,
             ForwardIterator, ForwardIterator, const T &,
             const T &)
{
    system.validate_dispatch();
}

void TestReplaceDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace(sys,
                    vec.begin(),
                    vec.begin(),
                    0,
                    0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceDispatchExplicit);


template<typename ForwardIterator, typename T>
void replace(my_tag,
             ForwardIterator first, ForwardIterator, const T &,
             const T &)
{
    *first = 13;
}

void TestReplaceDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::replace(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    0,
                    0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceDispatchImplicit);


template <typename T>
void TestReplace(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    T old_value = 0;
    T new_value = 1;

    thrust::replace(h_data.begin(), h_data.end(), old_value, new_value);
    thrust::replace(d_data.begin(), d_data.end(), old_value, new_value);

    ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestReplace);


template <class Vector>
void TestReplaceCopySimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] = 1; 
    data[1] = 2; 
    data[2] = 1;
    data[3] = 3; 
    data[4] = 2; 

    Vector dest(5);

    thrust::replace_copy(data.begin(), data.end(), dest.begin(), (T) 1, (T) 4);
    thrust::replace_copy(dest.begin(), dest.end(), dest.begin(), (T) 2, (T) 5);

    Vector result(5);
    result[0] = 4; 
    result[1] = 5; 
    result[2] = 4;
    result[3] = 3; 
    result[4] = 5; 

    ASSERT_EQUAL(dest, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceCopySimple);


template<typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(my_system &system,
                            InputIterator, InputIterator,
                            OutputIterator result,
                            const T &,
                            const T &)
{
    system.validate_dispatch();
    return result;
}

void TestReplaceCopyDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_copy(sys,
                         vec.begin(),
                         vec.begin(),
                         vec.begin(),
                         0,
                         0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceCopyDispatchExplicit);


template<typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(my_tag,
                            InputIterator, InputIterator,
                            OutputIterator result,
                            const T &,
                            const T &)
{
    *result = 13;
    return result;
}

void TestReplaceCopyDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::replace_copy(thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         0,
                         0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceCopyDispatchImplicit);


template <typename T>
void TestReplaceCopy(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;
    
    T old_value = 0;
    T new_value = 1;
    
    thrust::host_vector<T>   h_dest(n);
    thrust::device_vector<T> d_dest(n);

    thrust::replace_copy(h_data.begin(), h_data.end(), h_dest.begin(), old_value, new_value);
    thrust::replace_copy(d_data.begin(), d_data.end(), d_dest.begin(), old_value, new_value);

    ASSERT_ALMOST_EQUAL(h_data, d_data);
    ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopy);


template <typename T>
void TestReplaceCopyToDiscardIterator(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;
    
    T old_value = 0;
    T new_value = 1;

    thrust::discard_iterator<> h_result =
      thrust::replace_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), old_value, new_value);

    thrust::discard_iterator<> d_result =
      thrust::replace_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), old_value, new_value);

    thrust::discard_iterator<> reference(n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyToDiscardIterator);



template <typename T>
struct less_than_five
{
  __host__ __device__ bool operator()(const T &val) const {return val < 5;}
};

template <class Vector>
void TestReplaceIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  3; 
    data[2] =  4;
    data[3] =  6; 
    data[4] =  5; 

    thrust::replace_if(data.begin(), data.end(), less_than_five<T>(), (T) 0);

    Vector result(5);
    result[0] =  0; 
    result[1] =  0; 
    result[2] =  0;
    result[3] =  6; 
    result[4] =  5; 

    ASSERT_EQUAL(data, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceIfSimple);


template<typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_system &system,
                ForwardIterator, ForwardIterator,
                Predicate,
                const T &)
{
    system.validate_dispatch();
}

void TestReplaceIfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_if(sys,
                       vec.begin(),
                       vec.begin(),
                       0,
                       0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceIfDispatchExplicit);


template<typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_tag,
                ForwardIterator first, ForwardIterator,
                Predicate,
                const T &)
{
    *first = 13;
}

void TestReplaceIfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::replace_if(thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       0,
                       0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceIfDispatchImplicit);


template <class Vector>
void TestReplaceIfStencilSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  3; 
    data[2] =  4;
    data[3] =  6; 
    data[4] =  5; 

    Vector stencil(5);
    stencil[0] = 5;
    stencil[1] = 4;
    stencil[2] = 6;
    stencil[3] = 3;
    stencil[4] = 7;

    thrust::replace_if(data.begin(), data.end(), stencil.begin(), less_than_five<T>(), (T) 0);

    Vector result(5);
    result[0] =  1; 
    result[1] =  0; 
    result[2] =  4;
    result[3] =  0; 
    result[4] =  5; 

    ASSERT_EQUAL(data, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceIfStencilSimple);


template<typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(my_system &system,
                ForwardIterator, ForwardIterator,
                InputIterator,
                Predicate,
                const T &)
{
    system.validate_dispatch();
}

void TestReplaceIfStencilDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_if(sys,
                       vec.begin(),
                       vec.begin(),
                       vec.begin(),
                       0,
                       0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceIfStencilDispatchExplicit);


template<typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(my_tag,
                ForwardIterator first, ForwardIterator,
                InputIterator,
                Predicate,
                const T &)
{
    *first = 13;
}

void TestReplaceIfStencilDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::replace_if(thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       0,
                       0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceIfStencilDispatchImplicit);


template <typename T>
void TestReplaceIf(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::replace_if(h_data.begin(), h_data.end(), less_than_five<T>(), (T) 0);
    thrust::replace_if(d_data.begin(), d_data.end(), less_than_five<T>(), (T) 0);

    ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIf);


template <typename T>
void TestReplaceIfStencil(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T>   h_stencil = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_stencil = h_stencil;

    thrust::replace_if(h_data.begin(), h_data.end(), h_stencil.begin(), less_than_five<T>(), (T) 0);
    thrust::replace_if(d_data.begin(), d_data.end(), d_stencil.begin(), less_than_five<T>(), (T) 0);

    ASSERT_ALMOST_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceIfStencil);


template <class Vector>
void TestReplaceCopyIfSimple(void)
{
    typedef typename Vector::value_type T;
    
    Vector data(5);
    data[0] =  1; 
    data[1] =  3; 
    data[2] =  4;
    data[3] =  6; 
    data[4] =  5; 

    Vector dest(5);

    thrust::replace_copy_if(data.begin(), data.end(), dest.begin(), less_than_five<T>(), (T) 0);

    Vector result(5);
    result[0] =  0; 
    result[1] =  0; 
    result[2] =  0;
    result[3] =  6; 
    result[4] =  5; 

    ASSERT_EQUAL(dest, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceCopyIfSimple);


template<typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(my_system &system,
                               InputIterator, InputIterator,
                               OutputIterator result,
                               Predicate,
                               const T &)
{
    system.validate_dispatch();
    return result;
}

void TestReplaceCopyIfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_copy_if(sys,
                            vec.begin(),
                            vec.begin(),
                            vec.begin(),
                            0,
                            0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceCopyIfDispatchExplicit);


template<typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(my_tag,
                               InputIterator, InputIterator,
                               OutputIterator result,
                               Predicate,
                               const T &)
{
    *result = 13;
    return result;
}

void TestReplaceCopyIfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::replace_copy_if(thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            0,
                            0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceCopyIfDispatchImplicit);


template <class Vector>
void TestReplaceCopyIfStencilSimple(void)
{
    typedef typename Vector::value_type T;
    
    Vector data(5);
    data[0] =  1; 
    data[1] =  3; 
    data[2] =  4;
    data[3] =  6; 
    data[4] =  5; 

    Vector stencil(5);
    stencil[0] = 1;
    stencil[1] = 5;
    stencil[2] = 4;
    stencil[3] = 7;
    stencil[4] = 8;

    Vector dest(5);

    thrust::replace_copy_if(data.begin(), data.end(), stencil.begin(), dest.begin(), less_than_five<T>(), (T) 0);

    Vector result(5);
    result[0] =  0; 
    result[1] =  3; 
    result[2] =  0;
    result[3] =  6; 
    result[4] =  5; 

    ASSERT_EQUAL(dest, result);
}
DECLARE_VECTOR_UNITTEST(TestReplaceCopyIfStencilSimple);


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(my_system &system,
                               InputIterator1, InputIterator1,
                               InputIterator2,
                               OutputIterator result,
                               Predicate,
                               const T &)
{
    system.validate_dispatch();
    return result;
}


void TestReplaceCopyIfStencilDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_copy_if(sys,
                            vec.begin(),
                            vec.begin(),
                            vec.begin(),
                            vec.begin(),
                            0,
                            0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReplaceCopyIfStencilDispatchExplicit);


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(my_tag,
                               InputIterator1, InputIterator1,
                               InputIterator2,
                               OutputIterator result,
                               Predicate,
                               const T &)
{
    *result = 13;
    return result;
}

void TestReplaceCopyIfStencilDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::replace_copy_if(thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            0,
                            0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReplaceCopyIfStencilDispatchImplicit);


template <typename T>
void TestReplaceCopyIf(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T>   h_dest(n);
    thrust::device_vector<T> d_dest(n);

    thrust::replace_copy_if(h_data.begin(), h_data.end(), h_dest.begin(), less_than_five<T>(), T{0});
    thrust::replace_copy_if(d_data.begin(), d_data.end(), d_dest.begin(), less_than_five<T>(), T{0});

    ASSERT_ALMOST_EQUAL(h_data, d_data);
    ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIf);


template <typename T>
void TestReplaceCopyIfToDiscardIterator(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::discard_iterator<> h_result =
      thrust::replace_copy_if(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

    thrust::discard_iterator<> d_result =
      thrust::replace_copy_if(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

    thrust::discard_iterator<> reference(n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfToDiscardIterator);

template <typename T>
void TestReplaceCopyIfStencil(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T>   h_stencil = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_stencil = h_stencil;

    thrust::host_vector<T>   h_dest(n);
    thrust::device_vector<T> d_dest(n);

    thrust::replace_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_dest.begin(), less_than_five<T>(), T{0});
    thrust::replace_copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_dest.begin(), less_than_five<T>(), T{0});

    ASSERT_ALMOST_EQUAL(h_data, d_data);
    ASSERT_ALMOST_EQUAL(h_dest, d_dest);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfStencil);

template <typename T>
void TestReplaceCopyIfStencilToDiscardIterator(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T>   h_stencil = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_stencil = h_stencil;

    thrust::discard_iterator<> h_result =
      thrust::replace_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

    thrust::discard_iterator<> d_result =
      thrust::replace_copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

    thrust::discard_iterator<> reference(n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestReplaceCopyIfStencilToDiscardIterator);

