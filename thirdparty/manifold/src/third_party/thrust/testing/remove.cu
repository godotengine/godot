#include <unittest/unittest.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <stdexcept>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/retag.h>


template<typename T>
struct is_even
  : thrust::unary_function<T,bool>
{
    __host__ __device__
    bool operator()(T x) { return (static_cast<unsigned int>(x) & 1) == 0; }
};

template<typename T>
struct is_true
  : thrust::unary_function<T,bool>
{
    __host__ __device__
    bool operator()(T x) { return x ? true : false; }
};

template<typename Vector>
void TestRemoveSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1;
    data[1] =  2;
    data[2] =  1;
    data[3] =  3;
    data[4] =  2;

    typename Vector::iterator end = thrust::remove(data.begin(),
                                                    data.end(),
                                                    (T) 2);

    ASSERT_EQUAL(end - data.begin(), 3);

    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveSimple);


template<typename ForwardIterator,
         typename T>
ForwardIterator remove(my_system &system,
                       ForwardIterator first,
                       ForwardIterator,
                       const T &)
{
    system.validate_dispatch();
    return first;
}

void TestRemoveDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestRemoveDispatchExplicit);


template<typename ForwardIterator,
         typename T>
ForwardIterator remove(my_tag,
                       ForwardIterator first,
                       ForwardIterator,
                       const T &)
{
    *first = 13;
    return first;
}

void TestRemoveDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::remove(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.begin()),
                   0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestRemoveDispatchImplicit);


template<typename Vector>
void TestRemoveCopySimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1;
    data[1] =  2;
    data[2] =  1;
    data[3] =  3;
    data[4] =  2;

    Vector result(5);

    typename Vector::iterator end = thrust::remove_copy(data.begin(),
                                                        data.end(),
                                                        result.begin(),
                                                        (T) 2);

    ASSERT_EQUAL(end - result.begin(), 3);

    ASSERT_EQUAL(result[0], 1);
    ASSERT_EQUAL(result[1], 1);
    ASSERT_EQUAL(result[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveCopySimple);


template<typename InputIterator,
         typename OutputIterator,
         typename T>
OutputIterator remove_copy(my_system &system,
                           InputIterator,
                           InputIterator,
                           OutputIterator result,
                           const T &)
{
    system.validate_dispatch();
    return result;
}

void TestRemoveCopyDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_copy(sys,
                        vec.begin(),
                        vec.begin(),
                        vec.begin(),
                        0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestRemoveCopyDispatchExplicit);


template<typename InputIterator,
         typename OutputIterator,
         typename T>
OutputIterator remove_copy(my_tag,
                           InputIterator,
                           InputIterator,
                           OutputIterator result,
                           const T &)
{
    *result = 13;
    return result;
}

void TestRemoveCopyDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::remove_copy(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestRemoveCopyDispatchImplicit);


template<typename Vector>
void TestRemoveIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1;
    data[1] =  2;
    data[2] =  1;
    data[3] =  3;
    data[4] =  2;

    typename Vector::iterator end = thrust::remove_if(data.begin(),
                                                      data.end(),
                                                      is_even<T>());

    ASSERT_EQUAL(end - data.begin(), 3);

    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 3);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestRemoveIfSimple);


template<typename ForwardIterator,
         typename Predicate>
ForwardIterator remove_if(my_system &system,
                          ForwardIterator first,
                          ForwardIterator,
                          Predicate)
{
    system.validate_dispatch();
    return first;
}

void TestRemoveIfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_if(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestRemoveIfDispatchExplicit);


template<typename ForwardIterator,
         typename Predicate>
ForwardIterator remove_if(my_tag,
                          ForwardIterator first,
                          ForwardIterator,
                          Predicate)
{
    *first = 13;
    return first;
}

void TestRemoveIfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::remove_if(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestRemoveIfDispatchImplicit);


template<typename Vector>
void TestRemoveIfStencilSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1;
    data[1] =  2;
    data[2] =  1;
    data[3] =  3;
    data[4] =  2;

    Vector stencil(5);
    stencil[0] = 0;
    stencil[1] = 1;
    stencil[2] = 0;
    stencil[3] = 0;
    stencil[4] = 1;

    typename Vector::iterator end = thrust::remove_if(data.begin(),
                                                      data.end(),
                                                      stencil.begin(),
                                                      thrust::identity<T>());

    ASSERT_EQUAL(end - data.begin(), 3);

    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveIfStencilSimple);


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
ForwardIterator remove_if(my_system &system,
                          ForwardIterator first,
                          ForwardIterator,
                          InputIterator,
                          Predicate)
{
    system.validate_dispatch();
    return first;
}

void TestRemoveIfStencilDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_if(sys,
                      vec.begin(),
                      vec.begin(),
                      vec.begin(),
                      0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestRemoveIfStencilDispatchExplicit);


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
ForwardIterator remove_if(my_tag,
                          ForwardIterator first,
                          ForwardIterator,
                          InputIterator,
                          Predicate)
{
    *first = 13;
    return first;
}

void TestRemoveIfStencilDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::remove_if(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestRemoveIfStencilDispatchImplicit);


template<typename Vector>
void TestRemoveCopyIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1;
    data[1] =  2;
    data[2] =  1;
    data[3] =  3;
    data[4] =  2;

    Vector result(5);

    typename Vector::iterator end = thrust::remove_copy_if(data.begin(),
                                                           data.end(),
                                                           result.begin(),
                                                           is_even<T>());

    ASSERT_EQUAL(end - result.begin(), 3);

    ASSERT_EQUAL(result[0], 1);
    ASSERT_EQUAL(result[1], 1);
    ASSERT_EQUAL(result[2], 3);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestRemoveCopyIfSimple);


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
InputIterator remove_copy_if(my_system &system,
                             InputIterator first,
                             InputIterator,
                             OutputIterator,
                             Predicate)
{
    system.validate_dispatch();
    return first;
}

void TestRemoveCopyIfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_copy_if(sys,
                           vec.begin(),
                           vec.begin(),
                           vec.begin(),
                           0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestRemoveCopyIfDispatchExplicit);


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
InputIterator remove_copy_if(my_tag,
                             InputIterator first,
                             InputIterator,
                             OutputIterator,
                             Predicate)
{
    *first = 13;
    return first;
}

void TestRemoveCopyIfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::remove_copy_if(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestRemoveCopyIfDispatchImplicit);


template<typename Vector>
void TestRemoveCopyIfStencilSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1;
    data[1] =  2;
    data[2] =  1;
    data[3] =  3;
    data[4] =  2;

    Vector stencil(5);
    stencil[0] = 0;
    stencil[1] = 1;
    stencil[2] = 0;
    stencil[3] = 0;
    stencil[4] = 1;

    Vector result(5);

    typename Vector::iterator end = thrust::remove_copy_if(data.begin(),
                                                           data.end(),
                                                           stencil.begin(),
                                                           result.begin(),
                                                           thrust::identity<T>());

    ASSERT_EQUAL(end - result.begin(), 3);

    ASSERT_EQUAL(result[0], 1);
    ASSERT_EQUAL(result[1], 1);
    ASSERT_EQUAL(result[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveCopyIfStencilSimple);


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
OutputIterator remove_copy_if(my_system &system,
                              InputIterator1,
                              InputIterator1,
                              InputIterator2,
                              OutputIterator result,
                              Predicate)
{
    system.validate_dispatch();
    return result;
}

void TestRemoveCopyIfStencilDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_copy_if(sys,
                           vec.begin(),
                           vec.begin(),
                           vec.begin(),
                           vec.begin(),
                           0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestRemoveCopyIfStencilDispatchExplicit);


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
OutputIterator remove_copy_if(my_tag,
                              InputIterator1,
                              InputIterator1,
                              InputIterator2,
                              OutputIterator result,
                              Predicate)
{
    *result = 13;
    return result;
}

void TestRemoveCopyIfStencilDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::remove_copy_if(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestRemoveCopyIfStencilDispatchImplicit);


template<typename T>
void TestRemove(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t h_size = thrust::remove(h_data.begin(), h_data.end(), T(0)) - h_data.begin();
    size_t d_size = thrust::remove(d_data.begin(), d_data.end(), T(0)) - d_data.begin();

    ASSERT_EQUAL(h_size, d_size);

    h_data.resize(h_size);
    d_data.resize(d_size);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRemove);


template<typename T>
void TestRemoveIf(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), is_true<T>()) - h_data.begin();
    size_t d_size = thrust::remove_if(d_data.begin(), d_data.end(), is_true<T>()) - d_data.begin();

    ASSERT_EQUAL(h_size, d_size);

    h_data.resize(h_size);
    d_data.resize(d_size);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveIf);


template<typename T>
void TestRemoveIfStencil(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<bool>   h_stencil = unittest::random_integers<bool>(n);
    thrust::device_vector<bool> d_stencil = h_stencil;

    size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), h_stencil.begin(), is_true<T>()) - h_data.begin();
    size_t d_size = thrust::remove_if(d_data.begin(), d_data.end(), d_stencil.begin(), is_true<T>()) - d_data.begin();

    ASSERT_EQUAL(h_size, d_size);

    h_data.resize(h_size);
    d_data.resize(d_size);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveIfStencil);


template<typename T>
void TestRemoveCopy(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);

    size_t h_size = thrust::remove_copy(h_data.begin(), h_data.end(), h_result.begin(), T(0)) - h_result.begin();
    size_t d_size = thrust::remove_copy(d_data.begin(), d_data.end(), d_result.begin(), T(0)) - d_result.begin();

    ASSERT_EQUAL(h_size, d_size);

    h_result.resize(h_size);
    d_result.resize(d_size);

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopy);


template<typename T>
void TestRemoveCopyToDiscardIterator(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t num_zeros = thrust::count(h_data.begin(), h_data.end(), T(0));
    size_t num_nonzeros = h_data.size() - num_zeros;

    thrust::discard_iterator<> h_result =
      thrust::remove_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), T(0));

    thrust::discard_iterator<> d_result =
      thrust::remove_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), T(0));

    thrust::discard_iterator<> reference(num_nonzeros);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyToDiscardIterator);


template<typename T>
void TestRemoveCopyToDiscardIteratorZipped(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    size_t num_zeros = thrust::count(h_data.begin(), h_data.end(), T(0));
    size_t num_nonzeros = h_data.size() - num_zeros;

    typedef thrust::tuple<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<> >   Tuple1;
    typedef thrust::tuple<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<> > Tuple2;

    typedef thrust::zip_iterator<Tuple1> ZipIterator1;
    typedef thrust::zip_iterator<Tuple2> ZipIterator2;

    ZipIterator1 h_result =
      thrust::remove_copy(thrust::make_zip_iterator(thrust::make_tuple(h_data.begin(), h_data.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(h_data.end(), h_data.end())),
                          thrust::make_zip_iterator(thrust::make_tuple(h_output.begin(),thrust::make_discard_iterator())),
                          thrust::make_tuple(T(0),T(0)));

    ZipIterator2 d_result =
      thrust::remove_copy(thrust::make_zip_iterator(thrust::make_tuple(d_data.begin(), d_data.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_data.end(), d_data.end())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_output.begin(),thrust::make_discard_iterator())),
                          thrust::make_tuple(T(0),T(0)));

    thrust::discard_iterator<> reference(num_nonzeros);

    ASSERT_EQUAL(h_output, d_output);
    ASSERT_EQUAL_QUIET(reference, thrust::get<1>(h_result.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(reference, thrust::get<1>(d_result.get_iterator_tuple()));
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyToDiscardIteratorZipped);


template<typename T>
void TestRemoveCopyIf(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);

    size_t h_size = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>()) - h_result.begin();
    size_t d_size = thrust::remove_copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_true<T>()) - d_result.begin();

    ASSERT_EQUAL(h_size, d_size);

    h_result.resize(h_size);
    d_result.resize(d_size);

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIf);


template<typename T>
void TestRemoveCopyIfToDiscardIterator(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t num_false = thrust::count_if(h_data.begin(), h_data.end(), thrust::not1(is_true<T>()));

    thrust::discard_iterator<> h_result =
      thrust::remove_copy_if(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), is_true<T>());

    thrust::discard_iterator<> d_result =
      thrust::remove_copy_if(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), is_true<T>());

    thrust::discard_iterator<> reference(num_false);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIfToDiscardIterator);


template<typename T>
void TestRemoveCopyIfStencil(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<bool>   h_stencil = unittest::random_integers<bool>(n);
    thrust::device_vector<bool> d_stencil = h_stencil;

    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);

    size_t h_size = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), is_true<T>()) - h_result.begin();
    size_t d_size = thrust::remove_copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), is_true<T>()) - d_result.begin();

    ASSERT_EQUAL(h_size, d_size);

    h_result.resize(h_size);
    d_result.resize(d_size);

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIfStencil);


template<typename T>
void TestRemoveCopyIfStencilToDiscardIterator(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<bool>   h_stencil = unittest::random_integers<bool>(n);
    thrust::device_vector<bool> d_stencil = h_stencil;

    size_t num_false = thrust::count_if(h_stencil.begin(), h_stencil.end(), thrust::not1(is_true<T>()));

    thrust::discard_iterator<> h_result =
      thrust::remove_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), thrust::make_discard_iterator(), is_true<T>());

    thrust::discard_iterator<> d_result =
      thrust::remove_copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), thrust::make_discard_iterator(), is_true<T>());

    thrust::discard_iterator<> reference(num_false);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIfStencilToDiscardIterator);
