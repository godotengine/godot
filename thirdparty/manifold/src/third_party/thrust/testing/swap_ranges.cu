#include <unittest/unittest.h>
#include <thrust/swap.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>
#include <thrust/system/cpp/memory.h>


template<typename ForwardIterator1,
         typename ForwardIterator2>
ForwardIterator2 swap_ranges(my_system &system,
                             ForwardIterator1,
                             ForwardIterator1,
                             ForwardIterator2 first2)
{
    system.validate_dispatch();
    return first2;
}

void TestSwapRangesDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::swap_ranges(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSwapRangesDispatchExplicit);


template<typename ForwardIterator1,
         typename ForwardIterator2>
ForwardIterator2 swap_ranges(my_tag,
                             ForwardIterator1,
                             ForwardIterator1,
                             ForwardIterator2 first2)
{
    *first2 = 13;
    return first2;
}

void TestSwapRangesDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::swap_ranges(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSwapRangesDispatchImplicit);


template <class Vector>
void TestSwapRangesSimple(void)
{
    Vector v1(5);
    v1[0] = 0; v1[1] = 1; v1[2] = 2; v1[3] = 3; v1[4] = 4;

    Vector v2(5);
    v2[0] = 5; v2[1] = 6; v2[2] = 7; v2[3] = 8; v2[4] = 9;

    thrust::swap_ranges(v1.begin(), v1.end(), v2.begin());

    ASSERT_EQUAL(v1[0], 5);
    ASSERT_EQUAL(v1[1], 6);
    ASSERT_EQUAL(v1[2], 7);
    ASSERT_EQUAL(v1[3], 8);
    ASSERT_EQUAL(v1[4], 9);

    ASSERT_EQUAL(v2[0], 0);
    ASSERT_EQUAL(v2[1], 1);
    ASSERT_EQUAL(v2[2], 2);
    ASSERT_EQUAL(v2[3], 3);
    ASSERT_EQUAL(v2[4], 4);
}
DECLARE_VECTOR_UNITTEST(TestSwapRangesSimple);


template <typename T>
void TestSwapRanges(const size_t n)
{
    thrust::host_vector<T> a1 = unittest::random_integers<T>(n);
    thrust::host_vector<T> a2 = unittest::random_integers<T>(n);

    thrust::host_vector<T>    h1 = a1;
    thrust::host_vector<T>    h2 = a2;
    thrust::device_vector<T>  d1 = a1;
    thrust::device_vector<T>  d2 = a2;

    thrust::swap_ranges(h1.begin(), h1.end(), h2.begin());
    thrust::swap_ranges(d1.begin(), d1.end(), d2.begin());

    ASSERT_EQUAL(h1, a2);
    ASSERT_EQUAL(d1, a2);
    ASSERT_EQUAL(h2, a1);
    ASSERT_EQUAL(d2, a1);
}
DECLARE_VARIABLE_UNITTEST(TestSwapRanges);

#if (THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP)
void TestSwapRangesForcedIterator(void)
{
  thrust::device_vector<int> A(3, 0);
  thrust::device_vector<int> B(3, 1);

  thrust::swap_ranges(thrust::retag<thrust::cpp::tag>(A.begin()),
                      thrust::retag<thrust::cpp::tag>(A.end()),
                      thrust::retag<thrust::cpp::tag>(B.begin()));

  ASSERT_EQUAL(A[0], 1);
  ASSERT_EQUAL(A[1], 1);
  ASSERT_EQUAL(A[2], 1);
  ASSERT_EQUAL(B[0], 0);
  ASSERT_EQUAL(B[1], 0);
  ASSERT_EQUAL(B[2], 0);
}
DECLARE_UNITTEST(TestSwapRangesForcedIterator);
#endif

struct type_with_swap
{
  inline __host__ __device__
  type_with_swap()
    : m_x(), m_swapped(false)
  {}

  inline __host__ __device__
  type_with_swap(int x)
    : m_x(x), m_swapped(false)
  {}

  inline __host__ __device__
  type_with_swap(int x, bool s)
    : m_x(x), m_swapped(s)
  {}

  inline __host__ __device__
  type_with_swap(const type_with_swap &other)
    : m_x(other.m_x), m_swapped(other.m_swapped)
  {}

  inline __host__ __device__
  bool operator==(const type_with_swap &other) const
  {
    return m_x == other.m_x && m_swapped == other.m_swapped;
  }

#if THRUST_CPP_DIALECT >= 2011
  type_with_swap & operator=(const type_with_swap &) = default;
#endif

  int m_x;
  bool m_swapped;
};

inline __host__ __device__
void swap(type_with_swap &a, type_with_swap &b)
{
  thrust::swap(a.m_x, b.m_x);
  a.m_swapped = true;
  b.m_swapped = true;
}

void TestSwapRangesUserSwap(void)
{
  thrust::host_vector<type_with_swap> h_A(3, type_with_swap(0));
  thrust::host_vector<type_with_swap> h_B(3, type_with_swap(1));

  thrust::device_vector<type_with_swap> d_A = h_A;
  thrust::device_vector<type_with_swap> d_B = h_B;

  // check that nothing is yet swapped
  type_with_swap ref = type_with_swap(0, false);

  ASSERT_EQUAL_QUIET(ref, h_A[0]);
  ASSERT_EQUAL_QUIET(ref, h_A[1]);
  ASSERT_EQUAL_QUIET(ref, h_A[2]);

  ASSERT_EQUAL_QUIET(ref, d_A[0]);
  ASSERT_EQUAL_QUIET(ref, d_A[1]);
  ASSERT_EQUAL_QUIET(ref, d_A[2]);

  ref = type_with_swap(1, false);

  ASSERT_EQUAL_QUIET(ref, h_B[0]);
  ASSERT_EQUAL_QUIET(ref, h_B[1]);
  ASSERT_EQUAL_QUIET(ref, h_B[2]);

  ASSERT_EQUAL_QUIET(ref, d_B[0]);
  ASSERT_EQUAL_QUIET(ref, d_B[1]);
  ASSERT_EQUAL_QUIET(ref, d_B[2]);

  // swap the ranges

  thrust::swap_ranges(h_A.begin(), h_A.end(), h_B.begin());
  thrust::swap_ranges(d_A.begin(), d_A.end(), d_B.begin());

  // check that things were swapped
  ref = type_with_swap(1, true);

  ASSERT_EQUAL_QUIET(ref, h_A[0]);
  ASSERT_EQUAL_QUIET(ref, h_A[1]);
  ASSERT_EQUAL_QUIET(ref, h_A[2]);

  ASSERT_EQUAL_QUIET(ref, d_A[0]);
  ASSERT_EQUAL_QUIET(ref, d_A[1]);
  ASSERT_EQUAL_QUIET(ref, d_A[2]);

  ref = type_with_swap(0, true);

  ASSERT_EQUAL_QUIET(ref, h_B[0]);
  ASSERT_EQUAL_QUIET(ref, h_B[1]);
  ASSERT_EQUAL_QUIET(ref, h_B[2]);

  ASSERT_EQUAL_QUIET(ref, d_B[0]);
  ASSERT_EQUAL_QUIET(ref, d_B[1]);
  ASSERT_EQUAL_QUIET(ref, d_B[2]);
}
DECLARE_UNITTEST(TestSwapRangesUserSwap);

