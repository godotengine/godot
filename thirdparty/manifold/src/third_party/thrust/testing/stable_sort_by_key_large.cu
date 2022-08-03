#include <unittest/unittest.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

template <typename T>
struct less_div_10
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return ((int) lhs) / 10 < ((int) rhs) / 10;}
};

template <typename T>
struct greater_div_10
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return ((int) lhs) / 10 > ((int) rhs) / 10;}
};


template <typename T, unsigned int N>
void _TestStableSortByKeyWithLargeKeys(void)
{
    size_t n = (128 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_keys(n);
    thrust::host_vector<   unsigned int   > h_vals(n);

    for(size_t i = 0; i < n; i++)
    {
        const auto uint_i = static_cast<unsigned int>(i);
        const auto rand_int = unittest::generate_random_integer<T>()(uint_i);
        h_keys[i] = FixedVector<T,N>(rand_int);
        h_vals[i] = uint_i;
    }

    thrust::device_vector< FixedVector<T,N> > d_keys = h_keys;
    thrust::device_vector<   unsigned int   > d_vals = h_vals;
    
    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    ASSERT_EQUAL_QUIET(h_keys, d_keys);
    ASSERT_EQUAL_QUIET(h_vals, d_vals);
}

void TestStableSortByKeyWithLargeKeys(void)
{
    _TestStableSortByKeyWithLargeKeys<int,    4>();
    _TestStableSortByKeyWithLargeKeys<int,    8>();
    _TestStableSortByKeyWithLargeKeys<int,   16>();

// XXX these take too long to compile
//    _TestStableSortByKeyWithLargeKeys<int,   32>();
//    _TestStableSortByKeyWithLargeKeys<int,   64>();
//    _TestStableSortByKeyWithLargeKeys<int,  128>();
//    _TestStableSortByKeyWithLargeKeys<int,  256>();
//    _TestStableSortByKeyWithLargeKeys<int,  512>();
//    _TestStableSortByKeyWithLargeKeys<int, 1024>();
//    _TestStableSortByKeyWithLargeKeys<int, 2048>();
//    _TestStableSortByKeyWithLargeKeys<int, 4096>();
//    _TestStableSortByKeyWithLargeKeys<int, 8192>();
}
DECLARE_UNITTEST(TestStableSortByKeyWithLargeKeys);


template <typename T, unsigned int N>
void _TestStableSortByKeyWithLargeValues(void)
{
    size_t n = (128 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector<   unsigned int   > h_keys(n);
    thrust::host_vector< FixedVector<T,N> > h_vals(n);

    for(size_t i = 0; i < n; i++)
    {
        const auto uint_i = static_cast<unsigned int>(i);
        const auto rand_int =
          unittest::generate_random_integer<unsigned int>()(uint_i);
        h_keys[i] = rand_int;
        h_vals[i] = FixedVector<T,N>(static_cast<T>(i));
    }

    thrust::device_vector<   unsigned int   > d_keys = h_keys;
    thrust::device_vector< FixedVector<T,N> > d_vals = h_vals;
    
    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    ASSERT_EQUAL_QUIET(h_keys, d_keys);
    ASSERT_EQUAL_QUIET(h_vals, d_vals);

    // so cuda::stable_merge_sort_by_key() is called
    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), greater_div_10<unsigned int>());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), greater_div_10<unsigned int>());

    ASSERT_EQUAL_QUIET(h_keys, d_keys);
    ASSERT_EQUAL_QUIET(h_vals, d_vals);
}

void TestStableSortByKeyWithLargeValues(void)
{
    _TestStableSortByKeyWithLargeValues<int,    4>();
    _TestStableSortByKeyWithLargeValues<int,    8>();
    _TestStableSortByKeyWithLargeValues<int,   16>();
    
// XXX these take too long to compile
//    _TestStableSortByKeyWithLargeValues<int,   32>();
//    _TestStableSortByKeyWithLargeValues<int,   64>();
//    _TestStableSortByKeyWithLargeValues<int,  128>();
//    _TestStableSortByKeyWithLargeValues<int,  256>();
//    _TestStableSortByKeyWithLargeValues<int,  512>();
//    _TestStableSortByKeyWithLargeValues<int, 1024>();
//    _TestStableSortByKeyWithLargeValues<int, 2048>();
//    _TestStableSortByKeyWithLargeValues<int, 4096>();
//    _TestStableSortByKeyWithLargeValues<int, 8192>();
}
DECLARE_UNITTEST(TestStableSortByKeyWithLargeValues);


template <typename T, unsigned int N>
void _TestStableSortByKeyWithLargeKeysAndValues(void)
{
    size_t n = (128 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_keys(n);
    thrust::host_vector< FixedVector<T,N> > h_vals(n);

    for(size_t i = 0; i < n; i++)
    {
        const auto uint_i = static_cast<unsigned int>(i);
        const auto rand_int = unittest::generate_random_integer<T>()(uint_i);
        h_keys[i] = FixedVector<T,N>(rand_int);
        h_vals[i] = FixedVector<T,N>(static_cast<T>(i));
    }

    thrust::device_vector< FixedVector<T,N> > d_keys = h_keys;
    thrust::device_vector< FixedVector<T,N> > d_vals = h_vals;
    
    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    ASSERT_EQUAL_QUIET(h_keys, d_keys);
    ASSERT_EQUAL_QUIET(h_vals, d_vals);
}

void TestStableSortByKeyWithLargeKeysAndValues(void)
{
    _TestStableSortByKeyWithLargeKeysAndValues<int,    4>();
    _TestStableSortByKeyWithLargeKeysAndValues<int,    8>();
    _TestStableSortByKeyWithLargeKeysAndValues<int,   16>();

// XXX these take too long to compile
//    _TestStableSortByKeyWithLargeKeysAndValues<int,   32>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int,   64>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int,  128>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int,  256>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int,  512>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int, 1024>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int, 2048>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int, 4096>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int, 8192>();
}
DECLARE_UNITTEST(TestStableSortByKeyWithLargeKeysAndValues);

