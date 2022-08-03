#include <unittest/unittest.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

template <typename T>
  struct TestZipIteratorStableSort
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T>   h1 = unittest::random_integers<T>(n);
    host_vector<T>   h2 = unittest::random_integers<T>(n);
    
    device_vector<T> d1 = h1;
    device_vector<T> d2 = h2;
    
    // sort on host
    stable_sort( make_zip_iterator(make_tuple(h1.begin(), h2.begin())),
                 make_zip_iterator(make_tuple(h1.end(),   h2.end())) );

    // sort on device
    stable_sort( make_zip_iterator(make_tuple(d1.begin(), d2.begin())),
                 make_zip_iterator(make_tuple(d1.end(),   d2.end())) );
  
    ASSERT_EQUAL_QUIET(h1, d1);
    ASSERT_EQUAL_QUIET(h2, d2);
  }
};
VariableUnitTest<TestZipIteratorStableSort, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestZipIteratorStableSortInstance;

