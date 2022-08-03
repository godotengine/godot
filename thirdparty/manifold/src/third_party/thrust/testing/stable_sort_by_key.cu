#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>


template<typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(my_system &system, RandomAccessIterator1, RandomAccessIterator1, RandomAccessIterator2)
{
    system.validate_dispatch();
}

void TestStableSortByKeyDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::stable_sort_by_key(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestStableSortByKeyDispatchExplicit);


template<typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(my_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1, RandomAccessIterator2)
{
    *keys_first = 13;
}

void TestStableSortByKeyDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::stable_sort_by_key(thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestStableSortByKeyDispatchImplicit);

template <typename T>
struct less_div_10
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return ((int) lhs) / 10 < ((int) rhs) / 10;}
};


template <class Vector>
void InitializeSimpleStableKeyValueSortTest(Vector& unsorted_keys, Vector& unsorted_values,
                                            Vector& sorted_keys,   Vector& sorted_values)
{
    unsorted_keys.resize(9);   
    unsorted_values.resize(9);   
    unsorted_keys[0] = 25;   unsorted_values[0] = 0;   
    unsorted_keys[1] = 14;   unsorted_values[1] = 1; 
    unsorted_keys[2] = 35;   unsorted_values[2] = 2; 
    unsorted_keys[3] = 16;   unsorted_values[3] = 3; 
    unsorted_keys[4] = 26;   unsorted_values[4] = 4; 
    unsorted_keys[5] = 34;   unsorted_values[5] = 5; 
    unsorted_keys[6] = 36;   unsorted_values[6] = 6; 
    unsorted_keys[7] = 24;   unsorted_values[7] = 7; 
    unsorted_keys[8] = 15;   unsorted_values[8] = 8; 
    
    sorted_keys.resize(9);
    sorted_values.resize(9);
    sorted_keys[0] = 14;   sorted_values[0] = 1;    
    sorted_keys[1] = 16;   sorted_values[1] = 3; 
    sorted_keys[2] = 15;   sorted_values[2] = 8; 
    sorted_keys[3] = 25;   sorted_values[3] = 0; 
    sorted_keys[4] = 26;   sorted_values[4] = 4; 
    sorted_keys[5] = 24;   sorted_values[5] = 7; 
    sorted_keys[6] = 35;   sorted_values[6] = 2; 
    sorted_keys[7] = 34;   sorted_values[7] = 5; 
    sorted_keys[8] = 36;   sorted_values[8] = 6; 
}


template <class Vector>
void TestStableSortByKeySimple(void)
{
    typedef typename Vector::value_type T;

    Vector unsorted_keys, unsorted_values;
    Vector   sorted_keys,   sorted_values;

    InitializeSimpleStableKeyValueSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

    thrust::stable_sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin(), less_div_10<T>());

    ASSERT_EQUAL(unsorted_keys,   sorted_keys);
    ASSERT_EQUAL(unsorted_values, sorted_values);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestStableSortByKeySimple);


template <typename T>
struct TestStableSortByKey
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_keys = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_keys = h_keys;

        thrust::host_vector<T>   h_values = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_values = h_values;

        thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
        thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

        ASSERT_EQUAL(h_keys,   d_keys);
        ASSERT_EQUAL(h_values, d_values);
    }
};
VariableUnitTest<TestStableSortByKey, SignedIntegralTypes> TestStableSortByKeyInstance;


template <typename T>
struct TestStableSortByKeySemantics
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_keys = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_keys = h_keys;

        thrust::host_vector<T>   h_values = unittest::random_integers<T>(n);
        thrust::device_vector<T> d_values = h_values;

        thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), less_div_10<T>());
        thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), less_div_10<T>());

        ASSERT_EQUAL(h_keys,   d_keys);
        ASSERT_EQUAL(h_values, d_values);
    }
};
VariableUnitTest<TestStableSortByKeySemantics, unittest::type_list<unittest::uint8_t,unittest::uint16_t,unittest::uint32_t> > TestStableSortByKeySemanticsInstance;

