#include <unittest/unittest.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <algorithm>

THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

template <class Vector>
void TestFillSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    thrust::fill(v.begin() + 1, v.begin() + 4, (T) 7);

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 7);
    ASSERT_EQUAL(v[2], 7);
    ASSERT_EQUAL(v[3], 7);
    ASSERT_EQUAL(v[4], 4);

    thrust::fill(v.begin() + 0, v.begin() + 3, (T) 8);

    ASSERT_EQUAL(v[0], 8);
    ASSERT_EQUAL(v[1], 8);
    ASSERT_EQUAL(v[2], 8);
    ASSERT_EQUAL(v[3], 7);
    ASSERT_EQUAL(v[4], 4);

    thrust::fill(v.begin() + 2, v.end(), (T) 9);

    ASSERT_EQUAL(v[0], 8);
    ASSERT_EQUAL(v[1], 8);
    ASSERT_EQUAL(v[2], 9);
    ASSERT_EQUAL(v[3], 9);
    ASSERT_EQUAL(v[4], 9);

    thrust::fill(v.begin(), v.end(), (T) 1);

    ASSERT_EQUAL(v[0], 1);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 1);
    ASSERT_EQUAL(v[3], 1);
    ASSERT_EQUAL(v[4], 1);
}
DECLARE_VECTOR_UNITTEST(TestFillSimple);


void TestFillDiscardIterator(void)
{
    // there's no result to check because fill returns void
    thrust::fill(thrust::discard_iterator<thrust::host_system_tag>(),
                 thrust::discard_iterator<thrust::host_system_tag>(10),
                 13);

    thrust::fill(thrust::discard_iterator<thrust::device_system_tag>(),
                 thrust::discard_iterator<thrust::device_system_tag>(10),
                 13);
}
DECLARE_UNITTEST(TestFillDiscardIterator);


template <class Vector>
void TestFillMixedTypes(void)
{
    Vector v(4);

    thrust::fill(v.begin(), v.end(), bool(true));

    ASSERT_EQUAL(v[0], 1);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 1);
    ASSERT_EQUAL(v[3], 1);

    thrust::fill(v.begin(), v.end(), char(20));

    ASSERT_EQUAL(v[0], 20);
    ASSERT_EQUAL(v[1], 20);
    ASSERT_EQUAL(v[2], 20);
    ASSERT_EQUAL(v[3], 20);
}
DECLARE_VECTOR_UNITTEST(TestFillMixedTypes);


template <typename T>
void TestFill(size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::fill(h_data.begin() + std::min((size_t)1, n), h_data.begin() + std::min((size_t)3, n), (T) 0);
    thrust::fill(d_data.begin() + std::min((size_t)1, n), d_data.begin() + std::min((size_t)3, n), (T) 0);

    ASSERT_EQUAL(h_data, d_data);

    thrust::fill(h_data.begin() + std::min((size_t)117, n), h_data.begin() + std::min((size_t)367, n), (T) 1);
    thrust::fill(d_data.begin() + std::min((size_t)117, n), d_data.begin() + std::min((size_t)367, n), (T) 1);

    ASSERT_EQUAL(h_data, d_data);

    thrust::fill(h_data.begin() + std::min((size_t)8, n), h_data.begin() + std::min((size_t)259, n), (T) 2);
    thrust::fill(d_data.begin() + std::min((size_t)8, n), d_data.begin() + std::min((size_t)259, n), (T) 2);

    ASSERT_EQUAL(h_data, d_data);

    thrust::fill(h_data.begin() + std::min((size_t)3, n), h_data.end(), (T) 3);
    thrust::fill(d_data.begin() + std::min((size_t)3, n), d_data.end(), (T) 3);

    ASSERT_EQUAL(h_data, d_data);

    thrust::fill(h_data.begin(), h_data.end(), (T) 4);
    thrust::fill(d_data.begin(), d_data.end(), (T) 4);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestFill);

template <class Vector>
void TestFillNSimple(void)
{
    typedef typename Vector::value_type T;

    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    typename Vector::iterator iter = thrust::fill_n(v.begin() + 1, 3, (T) 7);

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 7);
    ASSERT_EQUAL(v[2], 7);
    ASSERT_EQUAL(v[3], 7);
    ASSERT_EQUAL(v[4], 4);
    ASSERT_EQUAL_QUIET(v.begin() + 4, iter);

    iter = thrust::fill_n(v.begin() + 0, 3, (T) 8);

    ASSERT_EQUAL(v[0], 8);
    ASSERT_EQUAL(v[1], 8);
    ASSERT_EQUAL(v[2], 8);
    ASSERT_EQUAL(v[3], 7);
    ASSERT_EQUAL(v[4], 4);
    ASSERT_EQUAL_QUIET(v.begin() + 3, iter);

    iter = thrust::fill_n(v.begin() + 2, 3, (T) 9);

    ASSERT_EQUAL(v[0], 8);
    ASSERT_EQUAL(v[1], 8);
    ASSERT_EQUAL(v[2], 9);
    ASSERT_EQUAL(v[3], 9);
    ASSERT_EQUAL(v[4], 9);
    ASSERT_EQUAL_QUIET(v.end(), iter);

    iter = thrust::fill_n(v.begin(), v.size(), (T) 1);

    ASSERT_EQUAL(v[0], 1);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 1);
    ASSERT_EQUAL(v[3], 1);
    ASSERT_EQUAL(v[4], 1);
    ASSERT_EQUAL_QUIET(v.end(), iter);
}
DECLARE_VECTOR_UNITTEST(TestFillNSimple);


void TestFillNDiscardIterator(void)
{
  thrust::discard_iterator<thrust::host_system_tag> h_result =
    thrust::fill_n(thrust::discard_iterator<thrust::host_system_tag>(),
                   10,
                   13);

  thrust::discard_iterator<thrust::device_system_tag> d_result =
    thrust::fill_n(thrust::discard_iterator<thrust::device_system_tag>(),
                   10,
                   13);

  thrust::discard_iterator<> reference(10);

  ASSERT_EQUAL_QUIET(reference, h_result);
  ASSERT_EQUAL_QUIET(reference, d_result);
}
DECLARE_UNITTEST(TestFillNDiscardIterator);


template <class Vector>
void TestFillNMixedTypes(void)
{
    Vector v(4);

    typename Vector::iterator iter = thrust::fill_n(v.begin(), v.size(), bool(true));

    ASSERT_EQUAL(v[0], 1);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 1);
    ASSERT_EQUAL(v[3], 1);
    ASSERT_EQUAL_QUIET(v.end(), iter);

    iter = thrust::fill_n(v.begin(), v.size(), char(20));

    ASSERT_EQUAL(v[0], 20);
    ASSERT_EQUAL(v[1], 20);
    ASSERT_EQUAL(v[2], 20);
    ASSERT_EQUAL(v[3], 20);
    ASSERT_EQUAL_QUIET(v.end(), iter);
}
DECLARE_VECTOR_UNITTEST(TestFillNMixedTypes);


template <typename T>
void TestFillN(size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t begin_offset = std::min<size_t>(1,n);
    thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t)3, n) - begin_offset, (T) 0);
    thrust::fill_n(d_data.begin() + begin_offset, std::min((size_t)3, n) - begin_offset, (T) 0);

    ASSERT_EQUAL(h_data, d_data);

    begin_offset = std::min<size_t>(117, n);
    thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t)367, n) - begin_offset, (T) 1);
    thrust::fill_n(d_data.begin() + begin_offset, std::min((size_t)367, n) - begin_offset, (T) 1);

    ASSERT_EQUAL(h_data, d_data);

    begin_offset = std::min<size_t>(8, n);
    thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t)259, n) - begin_offset, (T) 2);
    thrust::fill_n(d_data.begin() + begin_offset, std::min((size_t)259, n) - begin_offset, (T) 2);

    ASSERT_EQUAL(h_data, d_data);

    begin_offset = std::min<size_t>(3, n);
    thrust::fill_n(h_data.begin() + begin_offset, h_data.size() - begin_offset, (T) 3);
    thrust::fill_n(d_data.begin() + begin_offset, d_data.size() - begin_offset, (T) 3);

    ASSERT_EQUAL(h_data, d_data);

    thrust::fill_n(h_data.begin(), h_data.size(), (T) 4);
    thrust::fill_n(d_data.begin(), d_data.size(), (T) 4);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestFillN);


template <typename Vector>
void TestFillZipIterator(void)
{
    typedef typename Vector::value_type T;

    Vector v1(3,T(0));
    Vector v2(3,T(0));
    Vector v3(3,T(0));

    thrust::fill(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(),v2.begin(),v3.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(v1.end(),v2.end(),v3.end())),
                 thrust::tuple<T,T,T>(4,7,13));

    ASSERT_EQUAL(4,  v1[0]);
    ASSERT_EQUAL(4,  v1[1]);
    ASSERT_EQUAL(4,  v1[2]);
    ASSERT_EQUAL(7,  v2[0]);
    ASSERT_EQUAL(7,  v2[1]);
    ASSERT_EQUAL(7,  v2[2]);
    ASSERT_EQUAL(13, v3[0]);
    ASSERT_EQUAL(13, v3[1]);
    ASSERT_EQUAL(13, v3[2]);
};
DECLARE_VECTOR_UNITTEST(TestFillZipIterator);


void TestFillTuple(void)
{
    typedef int T;
    typedef thrust::tuple<T,T> Tuple;

    thrust::host_vector<Tuple>   h(3, Tuple(0,0));
    thrust::device_vector<Tuple> d(3, Tuple(0,0));

    thrust::fill(h.begin(), h.end(), Tuple(4,7));
    thrust::fill(d.begin(), d.end(), Tuple(4,7));

    ASSERT_EQUAL_QUIET(h, d);
};
DECLARE_UNITTEST(TestFillTuple);


struct TypeWithTrivialAssigment
{
  int x, y, z;
};

void TestFillWithTrivialAssignment(void)
{
    typedef TypeWithTrivialAssigment T;

    thrust::host_vector<T>   h(1);
    thrust::device_vector<T> d(1);

    ASSERT_EQUAL(h[0].x, 0);
    ASSERT_EQUAL(h[0].y, 0);
    ASSERT_EQUAL(h[0].z, 0);
    ASSERT_EQUAL(static_cast<T>(d[0]).x, 0);
    ASSERT_EQUAL(static_cast<T>(d[0]).y, 0);
    ASSERT_EQUAL(static_cast<T>(d[0]).z, 0);

    T val;
    val.x = 10;
    val.y = 20;
    val.z = -1;

    thrust::fill(h.begin(), h.end(), val);
    thrust::fill(d.begin(), d.end(), val);

    ASSERT_EQUAL(h[0].x, 10);
    ASSERT_EQUAL(h[0].y, 20);
    ASSERT_EQUAL(h[0].z, -1);
    ASSERT_EQUAL(static_cast<T>(d[0]).x, 10);
    ASSERT_EQUAL(static_cast<T>(d[0]).y, 20);
    ASSERT_EQUAL(static_cast<T>(d[0]).z, -1);
};
DECLARE_UNITTEST(TestFillWithTrivialAssignment);


struct TypeWithNonTrivialAssigment
{
  int x, y, z;

  __host__ __device__
  TypeWithNonTrivialAssigment() : x(0), y(0), z(0) {}

#if THRUST_CPP_DIALECT >= 2011
  TypeWithNonTrivialAssigment(const TypeWithNonTrivialAssigment &) = default;
#endif

  __host__ __device__
  TypeWithNonTrivialAssigment& operator=(const TypeWithNonTrivialAssigment& t)
  {
    x = t.x;
    y = t.y;
    z = t.x + t.y;
    return *this;
  }

  __host__ __device__
  bool operator==(const TypeWithNonTrivialAssigment& t) const
  {
    return x == t.x && y == t.y && z == t.z;
  }
};

void TestFillWithNonTrivialAssignment(void)
{
    typedef TypeWithNonTrivialAssigment T;

    thrust::host_vector<T>   h(1);
    thrust::device_vector<T> d(1);

    ASSERT_EQUAL(h[0].x, 0);
    ASSERT_EQUAL(h[0].y, 0);
    ASSERT_EQUAL(h[0].z, 0);
    ASSERT_EQUAL(static_cast<T>(d[0]).x, 0);
    ASSERT_EQUAL(static_cast<T>(d[0]).y, 0);
    ASSERT_EQUAL(static_cast<T>(d[0]).z, 0);

    T val;
    val.x = 10;
    val.y = 20;
    val.z = -1;

    thrust::fill(h.begin(), h.end(), val);
    thrust::fill(d.begin(), d.end(), val);

    ASSERT_EQUAL(h[0].x, 10);
    ASSERT_EQUAL(h[0].y, 20);
    ASSERT_EQUAL(h[0].z, 30);
    ASSERT_EQUAL(static_cast<T>(d[0]).x, 10);
    ASSERT_EQUAL(static_cast<T>(d[0]).y, 20);
    ASSERT_EQUAL(static_cast<T>(d[0]).z, 30);
};
DECLARE_UNITTEST(TestFillWithNonTrivialAssignment);


template<typename ForwardIterator, typename T>
void fill(my_system &system, ForwardIterator /*first*/, ForwardIterator, const T&)
{
    system.validate_dispatch();
}

void TestFillDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::fill(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestFillDispatchExplicit);


template<typename ForwardIterator, typename T>
void fill(my_tag, ForwardIterator first, ForwardIterator, const T&)
{
    *first = 13;
}

void TestFillDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::fill(thrust::retag<my_tag>(vec.begin()),
                 thrust::retag<my_tag>(vec.end()),
                 0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestFillDispatchImplicit);


template<typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(my_system &system, OutputIterator first, Size, const T&)
{
    system.validate_dispatch();
    return first;
}

void TestFillNDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::fill_n(sys, vec.begin(), vec.size(), 0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestFillNDispatchExplicit);


template<typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(my_tag, OutputIterator first, Size, const T&)
{
    *first = 13;
    return first;
}

void TestFillNDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::fill_n(thrust::retag<my_tag>(vec.begin()),
                   vec.size(),
                   0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestFillNDispatchImplicit);


THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END
