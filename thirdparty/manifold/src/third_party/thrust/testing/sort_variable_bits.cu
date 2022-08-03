#include <unittest/unittest.h>
#include <thrust/sort.h>

#include <algorithm>

using namespace unittest;

typedef unittest::type_list<
#if !(defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ <= 1))
// XXX GCC 4.1 miscompiles the char sorts with -O2 for some reason
                            unittest::uint8_t,
#endif
                            unittest::uint16_t,
                            unittest::uint32_t,
                            unittest::uint64_t> UnsignedIntegerTypes;

template <typename T>
struct TestSortVariableBits
{
  void operator()(const size_t n)
  {
    for(size_t num_bits = 0; num_bits < 8 * sizeof(T); num_bits += 3){

        thrust::host_vector<T>  h_keys = unittest::random_integers<T>(n);
   
        size_t mask = (1 << num_bits) - 1;
        for(size_t i = 0; i < n; i++)
            h_keys[i] &= mask;

        thrust::host_vector<T>   reference = h_keys;
        thrust::device_vector<T> d_keys    = h_keys;
    
        std::sort(reference.begin(), reference.end());

        thrust::sort(h_keys.begin(), h_keys.end());
        thrust::sort(d_keys.begin(), d_keys.end());
    
        ASSERT_EQUAL(reference, h_keys);
        ASSERT_EQUAL(h_keys, d_keys);
    }
  }
};
VariableUnitTest<TestSortVariableBits, UnsignedIntegerTypes> TestSortVariableBitsInstance;

