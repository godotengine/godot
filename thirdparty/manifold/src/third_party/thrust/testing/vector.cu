#include <unittest/unittest.h>

#include <thrust/detail/config.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>

#include <vector>
#include <list>
#include <limits>
#include <utility>

template <class Vector>
void TestVectorZeroSize(void)
{
    Vector v;
    ASSERT_EQUAL(v.size(), 0lu);
    ASSERT_EQUAL((v.begin() == v.end()), true);
}
DECLARE_VECTOR_UNITTEST(TestVectorZeroSize);


void TestVectorBool(void)
{
    thrust::host_vector<bool> h(3);
    thrust::device_vector<bool> d(3);

    h[0] = true; h[1] = false; h[2] = true;
    d[0] = true; d[1] = false; d[2] = true;

    ASSERT_EQUAL(h[0], true);
    ASSERT_EQUAL(h[1], false);
    ASSERT_EQUAL(h[2], true);

    ASSERT_EQUAL(d[0], true);
    ASSERT_EQUAL(d[1], false);
    ASSERT_EQUAL(d[2], true);
}
DECLARE_UNITTEST(TestVectorBool);


template <class Vector>
void TestVectorFrontBack(void)
{
    typedef typename Vector::value_type T;

    Vector v(3);
    v[0] = 0; v[1] = 1; v[2] = 2;

    ASSERT_EQUAL(v.front(), T(0));
    ASSERT_EQUAL(v.back(),  T(2));
}
DECLARE_VECTOR_UNITTEST(TestVectorFrontBack);


template <class Vector>
void TestVectorData(void)
{
    typedef typename Vector::pointer PointerT;
    typedef typename Vector::const_pointer PointerConstT;

    Vector v(3);
    v[0] = 0; v[1] = 1; v[2] = 2;

    ASSERT_EQUAL(0,          *v.data());
    ASSERT_EQUAL(1,          *(v.data() + 1));
    ASSERT_EQUAL(2,          *(v.data() + 2));
    ASSERT_EQUAL(PointerT(&v.front()),  v.data());
    ASSERT_EQUAL(PointerT(&*v.begin()), v.data());
    ASSERT_EQUAL(PointerT(&v[0]),       v.data());

    const Vector &c_v = v;

    ASSERT_EQUAL(0,            *c_v.data());
    ASSERT_EQUAL(1,            *(c_v.data() + 1));
    ASSERT_EQUAL(2,            *(c_v.data() + 2));
    ASSERT_EQUAL(PointerConstT(&c_v.front()),  c_v.data());
    ASSERT_EQUAL(PointerConstT(&*c_v.begin()), c_v.data());
    ASSERT_EQUAL(PointerConstT(&c_v[0]),       c_v.data());
}
DECLARE_VECTOR_UNITTEST(TestVectorData);


template <class Vector>
void TestVectorElementAssignment(void)
{
    Vector v(3);

    v[0] = 0; v[1] = 1; v[2] = 2;

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);

    v[0] = 10; v[1] = 11; v[2] = 12;

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(v[2], 12);

    Vector w(3);
    w[0] = v[0];
    w[1] = v[1];
    w[2] = v[2];

    ASSERT_EQUAL(v, w);
}
DECLARE_VECTOR_UNITTEST(TestVectorElementAssignment);


template <class Vector>
void TestVectorFromSTLVector(void)
{
    typedef typename Vector::value_type T;

    std::vector<T> stl_vector(3);
    stl_vector[0] = 0;
    stl_vector[1] = 1;
    stl_vector[2] = 2;

    thrust::host_vector<T> v(stl_vector);

    ASSERT_EQUAL(v.size(), 3lu);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);

    v = stl_vector;

    ASSERT_EQUAL(v.size(), 3lu);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestVectorFromSTLVector);


template <class Vector>
void TestVectorFillAssign(void)
{
    typedef typename Vector::value_type T;

    thrust::host_vector<T> v;
    v.assign(3, 13);

    ASSERT_EQUAL(v.size(), 3lu);
    ASSERT_EQUAL(v[0], 13);
    ASSERT_EQUAL(v[1], 13);
    ASSERT_EQUAL(v[2], 13);
}
DECLARE_VECTOR_UNITTEST(TestVectorFillAssign);


template <class Vector>
void TestVectorAssignFromSTLVector(void)
{
    typedef typename Vector::value_type T;

    std::vector<T> stl_vector(3);
    stl_vector[0] = 0;
    stl_vector[1] = 1;
    stl_vector[2] = 2;

    thrust::host_vector<T> v;
    v.assign(stl_vector.begin(), stl_vector.end());

    ASSERT_EQUAL(v.size(), 3lu);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestVectorAssignFromSTLVector);


template <class Vector>
void TestVectorFromBiDirectionalIterator(void)
{
    typedef typename Vector::value_type T;

    std::list<T> stl_list;
    stl_list.push_back(0);
    stl_list.push_back(1);
    stl_list.push_back(2);

    Vector v(stl_list.begin(), stl_list.end());

    ASSERT_EQUAL(v.size(), 3lu);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestVectorFromBiDirectionalIterator);


template <class Vector>
void TestVectorAssignFromBiDirectionalIterator(void)
{
    typedef typename Vector::value_type T;

    std::list<T> stl_list;
    stl_list.push_back(0);
    stl_list.push_back(1);
    stl_list.push_back(2);

    Vector v;
    v.assign(stl_list.begin(), stl_list.end());

    ASSERT_EQUAL(v.size(), 3lu);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestVectorAssignFromBiDirectionalIterator);


template <class Vector>
void TestVectorAssignFromHostVector(void)
{
    typedef typename Vector::value_type T;

    thrust::host_vector<T> h(3);
    h[0] = 0;
    h[1] = 1;
    h[2] = 2;

    Vector v;
    v.assign(h.begin(), h.end());

    ASSERT_EQUAL(v, h);
}
DECLARE_VECTOR_UNITTEST(TestVectorAssignFromHostVector);


template <class Vector>
void TestVectorToAndFromHostVector(void)
{
    typedef typename Vector::value_type T;

    thrust::host_vector<T> h(3);
    h[0] = 0;
    h[1] = 1;
    h[2] = 2;

    Vector v(h);

    ASSERT_EQUAL(v, h);

    THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING(v = v);

    ASSERT_EQUAL(v, h);

    v[0] = 10;
    v[1] = 11;
    v[2] = 12;

    ASSERT_EQUAL(h[0], 0);  ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(h[1], 1);  ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(h[2], 2);  ASSERT_EQUAL(v[2], 12);

    h = v;

    ASSERT_EQUAL(v, h);

    h[1] = 11;

    v = h;

    ASSERT_EQUAL(v, h);
}
DECLARE_VECTOR_UNITTEST(TestVectorToAndFromHostVector);


template <class Vector>
void TestVectorAssignFromDeviceVector(void)
{
    typedef typename Vector::value_type T;

    thrust::device_vector<T> d(3);
    d[0] = 0;
    d[1] = 1;
    d[2] = 2;

    Vector v;
    v.assign(d.begin(), d.end());

    ASSERT_EQUAL(v, d);
}
DECLARE_VECTOR_UNITTEST(TestVectorAssignFromDeviceVector);


template <class Vector>
void TestVectorToAndFromDeviceVector(void)
{
    typedef typename Vector::value_type T;

    thrust::device_vector<T> h(3);
    h[0] = 0;
    h[1] = 1;
    h[2] = 2;

    Vector v(h);

    ASSERT_EQUAL(v, h);

    THRUST_DISABLE_CLANG_SELF_ASSIGNMENT_WARNING(v = v);

    ASSERT_EQUAL(v, h);

    v[0] = 10;
    v[1] = 11;
    v[2] = 12;

    ASSERT_EQUAL(h[0], 0);  ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(h[1], 1);  ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(h[2], 2);  ASSERT_EQUAL(v[2], 12);

    h = v;

    ASSERT_EQUAL(v, h);

    h[1] = 11;

    v = h;

    ASSERT_EQUAL(v, h);
}
DECLARE_VECTOR_UNITTEST(TestVectorToAndFromDeviceVector);


template <class Vector>
void TestVectorWithInitialValue(void)
{
    typedef typename Vector::value_type T;

    const T init = 17;

    Vector v(3, init);

    ASSERT_EQUAL(v.size(), 3lu);
    ASSERT_EQUAL(v[0], init);
    ASSERT_EQUAL(v[1], init);
    ASSERT_EQUAL(v[2], init);
}
DECLARE_VECTOR_UNITTEST(TestVectorWithInitialValue);


template <class Vector>
void TestVectorSwap(void)
{
    Vector v(3);
    v[0] = 0; v[1] = 1; v[2] = 2;

    Vector u(3);
    u[0] = 10; u[1] = 11; u[2] = 12;

    v.swap(u);

    ASSERT_EQUAL(v[0], 10); ASSERT_EQUAL(u[0], 0);
    ASSERT_EQUAL(v[1], 11); ASSERT_EQUAL(u[1], 1);
    ASSERT_EQUAL(v[2], 12); ASSERT_EQUAL(u[2], 2);
}
DECLARE_VECTOR_UNITTEST(TestVectorSwap);


template <class Vector>
void TestVectorErasePosition(void)
{
    Vector v(5);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4;

    v.erase(v.begin() + 2);

    ASSERT_EQUAL(v.size(), 4lu);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 3);
    ASSERT_EQUAL(v[3], 4);

    v.erase(v.begin() + 0);

    ASSERT_EQUAL(v.size(), 3lu);
    ASSERT_EQUAL(v[0], 1);
    ASSERT_EQUAL(v[1], 3);
    ASSERT_EQUAL(v[2], 4);

    v.erase(v.begin() + 2);

    ASSERT_EQUAL(v.size(), 2lu);
    ASSERT_EQUAL(v[0], 1);
    ASSERT_EQUAL(v[1], 3);

    v.erase(v.begin() + 1);

    ASSERT_EQUAL(v.size(), 1lu);
    ASSERT_EQUAL(v[0], 1);

    v.erase(v.begin() + 0);

    ASSERT_EQUAL(v.size(), 0lu);
}
DECLARE_VECTOR_UNITTEST(TestVectorErasePosition);


template <class Vector>
void TestVectorEraseRange(void)
{
    Vector v(6);
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4; v[5] = 5;

    v.erase(v.begin() + 1, v.begin() + 3);

    ASSERT_EQUAL(v.size(), 4lu);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 3);
    ASSERT_EQUAL(v[2], 4);
    ASSERT_EQUAL(v[3], 5);

    v.erase(v.begin() + 2, v.end());

    ASSERT_EQUAL(v.size(), 2lu);
    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 3);

    v.erase(v.begin() + 0, v.begin() + 1);

    ASSERT_EQUAL(v.size(), 1lu);
    ASSERT_EQUAL(v[0], 3);

    v.erase(v.begin(), v.end());

    ASSERT_EQUAL(v.size(), 0lu);
}
DECLARE_VECTOR_UNITTEST(TestVectorEraseRange);


void TestVectorEquality(void)
{
    thrust::host_vector<int> h_a(3);
    thrust::host_vector<int> h_b(3);
    thrust::host_vector<int> h_c(3);
    h_a[0] = 0;    h_a[1] = 1;    h_a[2] = 2;
    h_b[0] = 0;    h_b[1] = 1;    h_b[2] = 3;
    h_b[0] = 0;    h_b[1] = 1;

    thrust::device_vector<int> d_a(3);
    thrust::device_vector<int> d_b(3);
    thrust::device_vector<int> d_c(3);
    d_a[0] = 0;    d_a[1] = 1;    d_a[2] = 2;
    d_b[0] = 0;    d_b[1] = 1;    d_b[2] = 3;
    d_b[0] = 0;    d_b[1] = 1;

    std::vector<int> s_a(3);
    std::vector<int> s_b(3);
    std::vector<int> s_c(3);
    s_a[0] = 0;    s_a[1] = 1;    s_a[2] = 2;
    s_b[0] = 0;    s_b[1] = 1;    s_b[2] = 3;
    s_b[0] = 0;    s_b[1] = 1;

    ASSERT_EQUAL((h_a == h_a), true); ASSERT_EQUAL((h_a == d_a), true); ASSERT_EQUAL((d_a == h_a), true);  ASSERT_EQUAL((d_a == d_a), true);
    ASSERT_EQUAL((h_b == h_b), true); ASSERT_EQUAL((h_b == d_b), true); ASSERT_EQUAL((d_b == h_b), true);  ASSERT_EQUAL((d_b == d_b), true);
    ASSERT_EQUAL((h_c == h_c), true); ASSERT_EQUAL((h_c == d_c), true); ASSERT_EQUAL((d_c == h_c), true);  ASSERT_EQUAL((d_c == d_c), true);

    // test vector vs device_vector
    ASSERT_EQUAL((s_a == d_a), true); ASSERT_EQUAL((d_a == s_a), true);
    ASSERT_EQUAL((s_b == d_b), true); ASSERT_EQUAL((d_b == s_b), true);
    ASSERT_EQUAL((s_c == d_c), true); ASSERT_EQUAL((d_c == s_c), true);

    // test vector vs host_vector
    ASSERT_EQUAL((s_a == h_a), true); ASSERT_EQUAL((h_a == s_a), true);
    ASSERT_EQUAL((s_b == h_b), true); ASSERT_EQUAL((h_b == s_b), true);
    ASSERT_EQUAL((s_c == h_c), true); ASSERT_EQUAL((h_c == s_c), true);

    ASSERT_EQUAL((h_a == h_b), false); ASSERT_EQUAL((h_a == d_b), false); ASSERT_EQUAL((d_a == h_b), false); ASSERT_EQUAL((d_a == d_b), false);
    ASSERT_EQUAL((h_b == h_a), false); ASSERT_EQUAL((h_b == d_a), false); ASSERT_EQUAL((d_b == h_a), false); ASSERT_EQUAL((d_b == d_a), false);
    ASSERT_EQUAL((h_a == h_c), false); ASSERT_EQUAL((h_a == d_c), false); ASSERT_EQUAL((d_a == h_c), false); ASSERT_EQUAL((d_a == d_c), false);
    ASSERT_EQUAL((h_c == h_a), false); ASSERT_EQUAL((h_c == d_a), false); ASSERT_EQUAL((d_c == h_a), false); ASSERT_EQUAL((d_c == d_a), false);
    ASSERT_EQUAL((h_b == h_c), false); ASSERT_EQUAL((h_b == d_c), false); ASSERT_EQUAL((d_b == h_c), false); ASSERT_EQUAL((d_b == d_c), false);
    ASSERT_EQUAL((h_c == h_b), false); ASSERT_EQUAL((h_c == d_b), false); ASSERT_EQUAL((d_c == h_b), false); ASSERT_EQUAL((d_c == d_b), false);

    // test vector vs device_vector
    ASSERT_EQUAL((s_a == d_b), false); ASSERT_EQUAL((d_a == s_b), false);
    ASSERT_EQUAL((s_b == d_a), false); ASSERT_EQUAL((d_b == s_a), false);
    ASSERT_EQUAL((s_a == d_c), false); ASSERT_EQUAL((d_a == s_c), false);
    ASSERT_EQUAL((s_c == d_a), false); ASSERT_EQUAL((d_c == s_a), false);
    ASSERT_EQUAL((s_b == d_c), false); ASSERT_EQUAL((d_b == s_c), false);
    ASSERT_EQUAL((s_c == d_b), false); ASSERT_EQUAL((d_c == s_b), false);

    // test vector vs host_vector
    ASSERT_EQUAL((s_a == h_b), false); ASSERT_EQUAL((h_a == s_b), false);
    ASSERT_EQUAL((s_b == h_a), false); ASSERT_EQUAL((h_b == s_a), false);
    ASSERT_EQUAL((s_a == h_c), false); ASSERT_EQUAL((h_a == s_c), false);
    ASSERT_EQUAL((s_c == h_a), false); ASSERT_EQUAL((h_c == s_a), false);
    ASSERT_EQUAL((s_b == h_c), false); ASSERT_EQUAL((h_b == s_c), false);
    ASSERT_EQUAL((s_c == h_b), false); ASSERT_EQUAL((h_c == s_b), false);
}
DECLARE_UNITTEST(TestVectorEquality);

void TestVectorInequality(void)
{
    thrust::host_vector<int> h_a(3);
    thrust::host_vector<int> h_b(3);
    thrust::host_vector<int> h_c(3);
    h_a[0] = 0;    h_a[1] = 1;    h_a[2] = 2;
    h_b[0] = 0;    h_b[1] = 1;    h_b[2] = 3;
    h_b[0] = 0;    h_b[1] = 1;

    thrust::device_vector<int> d_a(3);
    thrust::device_vector<int> d_b(3);
    thrust::device_vector<int> d_c(3);
    d_a[0] = 0;    d_a[1] = 1;    d_a[2] = 2;
    d_b[0] = 0;    d_b[1] = 1;    d_b[2] = 3;
    d_b[0] = 0;    d_b[1] = 1;

    std::vector<int> s_a(3);
    std::vector<int> s_b(3);
    std::vector<int> s_c(3);
    s_a[0] = 0;    s_a[1] = 1;    s_a[2] = 2;
    s_b[0] = 0;    s_b[1] = 1;    s_b[2] = 3;
    s_b[0] = 0;    s_b[1] = 1;

    ASSERT_EQUAL((h_a != h_a), false); ASSERT_EQUAL((h_a != d_a), false); ASSERT_EQUAL((d_a != h_a), false);  ASSERT_EQUAL((d_a != d_a), false);
    ASSERT_EQUAL((h_b != h_b), false); ASSERT_EQUAL((h_b != d_b), false); ASSERT_EQUAL((d_b != h_b), false);  ASSERT_EQUAL((d_b != d_b), false);
    ASSERT_EQUAL((h_c != h_c), false); ASSERT_EQUAL((h_c != d_c), false); ASSERT_EQUAL((d_c != h_c), false);  ASSERT_EQUAL((d_c != d_c), false);

    // test vector vs device_vector
    ASSERT_EQUAL((s_a != d_a), false); ASSERT_EQUAL((d_a != s_a), false);
    ASSERT_EQUAL((s_b != d_b), false); ASSERT_EQUAL((d_b != s_b), false);
    ASSERT_EQUAL((s_c != d_c), false); ASSERT_EQUAL((d_c != s_c), false);

    // test vector vs host_vector
    ASSERT_EQUAL((s_a != h_a), false); ASSERT_EQUAL((h_a != s_a), false);
    ASSERT_EQUAL((s_b != h_b), false); ASSERT_EQUAL((h_b != s_b), false);
    ASSERT_EQUAL((s_c != h_c), false); ASSERT_EQUAL((h_c != s_c), false);

    ASSERT_EQUAL((h_a != h_b), true); ASSERT_EQUAL((h_a != d_b), true); ASSERT_EQUAL((d_a != h_b), true); ASSERT_EQUAL((d_a != d_b), true);
    ASSERT_EQUAL((h_b != h_a), true); ASSERT_EQUAL((h_b != d_a), true); ASSERT_EQUAL((d_b != h_a), true); ASSERT_EQUAL((d_b != d_a), true);
    ASSERT_EQUAL((h_a != h_c), true); ASSERT_EQUAL((h_a != d_c), true); ASSERT_EQUAL((d_a != h_c), true); ASSERT_EQUAL((d_a != d_c), true);
    ASSERT_EQUAL((h_c != h_a), true); ASSERT_EQUAL((h_c != d_a), true); ASSERT_EQUAL((d_c != h_a), true); ASSERT_EQUAL((d_c != d_a), true);
    ASSERT_EQUAL((h_b != h_c), true); ASSERT_EQUAL((h_b != d_c), true); ASSERT_EQUAL((d_b != h_c), true); ASSERT_EQUAL((d_b != d_c), true);
    ASSERT_EQUAL((h_c != h_b), true); ASSERT_EQUAL((h_c != d_b), true); ASSERT_EQUAL((d_c != h_b), true); ASSERT_EQUAL((d_c != d_b), true);

    // test vector vs device_vector
    ASSERT_EQUAL((s_a != d_b), true); ASSERT_EQUAL((d_a != s_b), true);
    ASSERT_EQUAL((s_b != d_a), true); ASSERT_EQUAL((d_b != s_a), true);
    ASSERT_EQUAL((s_a != d_c), true); ASSERT_EQUAL((d_a != s_c), true);
    ASSERT_EQUAL((s_c != d_a), true); ASSERT_EQUAL((d_c != s_a), true);
    ASSERT_EQUAL((s_b != d_c), true); ASSERT_EQUAL((d_b != s_c), true);
    ASSERT_EQUAL((s_c != d_b), true); ASSERT_EQUAL((d_c != s_b), true);

    // test vector vs host_vector
    ASSERT_EQUAL((s_a != h_b), true); ASSERT_EQUAL((h_a != s_b), true);
    ASSERT_EQUAL((s_b != h_a), true); ASSERT_EQUAL((h_b != s_a), true);
    ASSERT_EQUAL((s_a != h_c), true); ASSERT_EQUAL((h_a != s_c), true);
    ASSERT_EQUAL((s_c != h_a), true); ASSERT_EQUAL((h_c != s_a), true);
    ASSERT_EQUAL((s_b != h_c), true); ASSERT_EQUAL((h_b != s_c), true);
    ASSERT_EQUAL((s_c != h_b), true); ASSERT_EQUAL((h_c != s_b), true);
}
DECLARE_UNITTEST(TestVectorInequality);


template <class Vector>
void TestVectorResizing(void)
{
    Vector v;

    v.resize(3);

    ASSERT_EQUAL(v.size(), 3lu);

    v[0] = 0; v[1] = 1; v[2] = 2;

    v.resize(5);

    ASSERT_EQUAL(v.size(), 5lu);

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);

    v[3] = 3; v[4] = 4;

    v.resize(4);

    ASSERT_EQUAL(v.size(), 4lu);

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
    ASSERT_EQUAL(v[3], 3);

    v.resize(0);

    ASSERT_EQUAL(v.size(), 0lu);

// TODO remove this WAR
#if defined(__CUDACC__) && CUDART_VERSION==3000
    // depending on sizeof(T), we will receive one
    // of two possible exceptions
    try
    {
      v.resize(std::numeric_limits<size_t>::max());
    }
    catch(std::length_error e) {}
    catch(std::bad_alloc e)
    {
      // reset the CUDA error
      cudaGetLastError();
    } // end catch
#endif // defined(__CUDACC__) && CUDART_VERSION==3000

    ASSERT_EQUAL(v.size(), 0lu);
}
DECLARE_VECTOR_UNITTEST(TestVectorResizing);



template <class Vector>
void TestVectorReserving(void)
{
    Vector v;

    v.reserve(3);

    ASSERT_GEQUAL(v.capacity(), 3lu);

    size_t old_capacity = v.capacity();

    v.reserve(0);

    ASSERT_EQUAL(v.capacity(), old_capacity);

// TODO remove this WAR
#if defined(__CUDACC__) && CUDART_VERSION==3000
    try
    {
      v.reserve(std::numeric_limits<size_t>::max());
    }
    catch(std::length_error e) {}
    catch(std::bad_alloc e) {}
#endif // defined(__CUDACC__) && CUDART_VERSION==3000

    ASSERT_EQUAL(v.capacity(), old_capacity);
}
DECLARE_VECTOR_UNITTEST(TestVectorReserving)



template <class Vector>
void TestVectorUninitialisedCopy(void)
{
    thrust::device_vector<int> v;
    std::vector<int> std_vector;

    v = std_vector;

    ASSERT_EQUAL(v.size(), static_cast<size_t>(0));
}
DECLARE_VECTOR_UNITTEST(TestVectorUninitialisedCopy);


template <class Vector>
void TestVectorShrinkToFit(void)
{
    typedef typename Vector::value_type T;

    Vector v;

    v.reserve(200);

    ASSERT_GEQUAL(v.capacity(), 200lu);

    v.push_back(1);
    v.push_back(2);
    v.push_back(3);

    v.shrink_to_fit();

    ASSERT_EQUAL(T(1), v[0]);
    ASSERT_EQUAL(T(2), v[1]);
    ASSERT_EQUAL(T(3), v[2]);
    ASSERT_EQUAL(3lu, v.size());
    ASSERT_EQUAL(3lu, v.capacity());
}
DECLARE_VECTOR_UNITTEST(TestVectorShrinkToFit)

template <int N>
struct LargeStruct
{
  int data[N];

  __host__ __device__
  bool operator==(const LargeStruct & ls) const
  {
    for (int i = 0; i < N; i++)
      if (data[i] != ls.data[i])
        return false;
    return true;
  }
};

void TestVectorContainingLargeType(void)
{
    // Thrust issue #5
    // http://code.google.com/p/thrust/issues/detail?id=5
    const static int N = 100;
    typedef LargeStruct<N> T;

    thrust::device_vector<T> dv1;
    thrust::host_vector<T>   hv1;

    ASSERT_EQUAL_QUIET(dv1, hv1);

    thrust::device_vector<T> dv2(20);
    thrust::host_vector<T>   hv2(20);

    ASSERT_EQUAL_QUIET(dv2, hv2);

    // initialize tofirst element to something nonzero
    T ls;

    for (int i = 0; i < N; i++)
      ls.data[i] = i;

    thrust::device_vector<T> dv3(20, ls);
    thrust::host_vector<T>   hv3(20, ls);

    ASSERT_EQUAL_QUIET(dv3, hv3);

    // change first element
    ls.data[0] = -13;

    dv3[2] = ls;
    hv3[2] = ls;

    ASSERT_EQUAL_QUIET(dv3, hv3);
}
DECLARE_UNITTEST(TestVectorContainingLargeType);


template <typename Vector>
void TestVectorReversed(void)
{
  Vector v(3);
  v[0] = 0; v[1] = 1; v[2] = 2;

  ASSERT_EQUAL(3, v.rend() - v.rbegin());
  ASSERT_EQUAL(3, static_cast<const Vector&>(v).rend() - static_cast<const Vector&>(v).rbegin());
  ASSERT_EQUAL(3, v.crend() - v.crbegin());

  ASSERT_EQUAL(2, *v.rbegin());
  ASSERT_EQUAL(2, *static_cast<const Vector&>(v).rbegin());
  ASSERT_EQUAL(2, *v.crbegin());

  ASSERT_EQUAL(1, *(v.rbegin() + 1));
  ASSERT_EQUAL(0, *(v.rbegin() + 2));

  ASSERT_EQUAL(0, *(v.rend() - 1));
  ASSERT_EQUAL(1, *(v.rend() - 2));
}
DECLARE_VECTOR_UNITTEST(TestVectorReversed);

#if THRUST_CPP_DIALECT >= 2011
  template <class Vector>
  void TestVectorMove(void)
  {
    //test move construction
    Vector v1(3);
    v1[0] = 0; v1[1] = 1; v1[2] = 2;

    const auto ptr1 = v1.data();
    const auto size1 = v1.size();

    Vector v2(std::move(v1));
    const auto ptr2 = v2.data();
    const auto size2 = v2.size();

    // ensure v1 was left empty
    ASSERT_EQUAL(true, v1.empty());

    // ensure v2 received the data from before
    ASSERT_EQUAL(v2[0], 0);
    ASSERT_EQUAL(v2[1], 1);
    ASSERT_EQUAL(v2[2], 2);
    ASSERT_EQUAL(size1, size2);

    // ensure v2 received the pointer from before
    ASSERT_EQUAL(ptr1, ptr2);

    //test move assignment
    Vector v3(3);
    v3[0] = 3; v3[1] = 4; v3[2] = 5;

    const auto ptr3 = v3.data();
    const auto size3 = v3.size();

    v2 = std::move(v3);
    const auto ptr4 = v2.data();
    const auto size4 = v2.size();

    // ensure v3 was left empty
    ASSERT_EQUAL(true, v3.empty());

    // ensure v2 received the data from before
    ASSERT_EQUAL(v2[0], 3);
    ASSERT_EQUAL(v2[1], 4);
    ASSERT_EQUAL(v2[2], 5);
    ASSERT_EQUAL(size3, size4);

    // ensure v2 received the pointer from before
    ASSERT_EQUAL(ptr3, ptr4);
  }
  DECLARE_VECTOR_UNITTEST(TestVectorMove);
#endif

