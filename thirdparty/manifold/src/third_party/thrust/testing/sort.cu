#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>


template<typename RandomAccessIterator>
void sort(my_system &system, RandomAccessIterator, RandomAccessIterator)
{
  system.validate_dispatch();
}

void TestSortDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::sort(sys, vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSortDispatchExplicit);


template<typename RandomAccessIterator>
void sort(my_tag, RandomAccessIterator first, RandomAccessIterator)
{
  *first = 13;
}

void TestSortDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::sort(thrust::retag<my_tag>(vec.begin()),
               thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSortDispatchImplicit);

template <class Vector>
void InitializeSimpleKeySortTest(Vector& unsorted_keys, Vector& sorted_keys)
{
    unsorted_keys.resize(7);
    unsorted_keys[0] = 1; 
    unsorted_keys[1] = 3; 
    unsorted_keys[2] = 6;
    unsorted_keys[3] = 5;
    unsorted_keys[4] = 2;
    unsorted_keys[5] = 0;
    unsorted_keys[6] = 4;

    sorted_keys.resize(7); 
    sorted_keys[0] = 0; 
    sorted_keys[1] = 1; 
    sorted_keys[2] = 2;
    sorted_keys[3] = 3;
    sorted_keys[4] = 4;
    sorted_keys[5] = 5;
    sorted_keys[6] = 6;
}

template <class Vector>
void TestSortSimple(void)
{
    Vector unsorted_keys;
    Vector   sorted_keys;

    InitializeSimpleKeySortTest(unsorted_keys, sorted_keys);

    thrust::sort(unsorted_keys.begin(), unsorted_keys.end());

    ASSERT_EQUAL(unsorted_keys, sorted_keys);
}
DECLARE_VECTOR_UNITTEST(TestSortSimple);


template <typename T>
void TestSortAscendingKey(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::sort(h_data.begin(), h_data.end(), thrust::less<T>());
    thrust::sort(d_data.begin(), d_data.end(), thrust::less<T>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestSortAscendingKey);

void TestSortDescendingKey(void)
{
    const size_t n = 10027;

    thrust::host_vector<int>   h_data = unittest::random_integers<int>(n);
    thrust::device_vector<int> d_data = h_data;

    thrust::sort(h_data.begin(), h_data.end(), thrust::greater<int>());
    thrust::sort(d_data.begin(), d_data.end(), thrust::greater<int>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_UNITTEST(TestSortDescendingKey);


void TestSortBool(void)
{
    const size_t n = 10027;

    thrust::host_vector<bool>   h_data = unittest::random_integers<bool>(n);
    thrust::device_vector<bool> d_data = h_data;

    thrust::sort(h_data.begin(), h_data.end());
    thrust::sort(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_UNITTEST(TestSortBool);


void TestSortBoolDescending(void)
{
    const size_t n = 10027;

    thrust::host_vector<bool>   h_data = unittest::random_integers<bool>(n);
    thrust::device_vector<bool> d_data = h_data;

    thrust::sort(h_data.begin(), h_data.end(), thrust::greater<bool>());
    thrust::sort(d_data.begin(), d_data.end(), thrust::greater<bool>());

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_UNITTEST(TestSortBoolDescending);


