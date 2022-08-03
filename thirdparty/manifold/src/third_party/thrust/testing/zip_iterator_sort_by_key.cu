#include <unittest/unittest.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

template <typename T>
  struct TestZipIteratorStableSortByKey
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T>   h1 = unittest::random_integers<T>(n);
    host_vector<T>   h2 = unittest::random_integers<T>(n);
    host_vector<T>   h3 = unittest::random_integers<T>(n);
    host_vector<T>   h4 = unittest::random_integers<T>(n);
    
    device_vector<T> d1 = h1;
    device_vector<T> d2 = h2;
    device_vector<T> d3 = h3;
    device_vector<T> d4 = h4;
    
    // sort with (tuple, scalar)
    stable_sort_by_key( make_zip_iterator(make_tuple(h1.begin(), h2.begin())),
                        make_zip_iterator(make_tuple(h1.end(),   h2.end())),
                        h3.begin() );
    stable_sort_by_key( make_zip_iterator(make_tuple(d1.begin(), d2.begin())),
                        make_zip_iterator(make_tuple(d1.end(),   d2.end())),
                        d3.begin() );
    
    ASSERT_EQUAL_QUIET(h1, d1);
    ASSERT_EQUAL_QUIET(h2, d2);
    ASSERT_EQUAL_QUIET(h3, d3);
    ASSERT_EQUAL_QUIET(h4, d4);
    
    // sort with (scalar, tuple)
    stable_sort_by_key( h1.begin(),
                        h1.end(),
                        make_zip_iterator(make_tuple(h3.begin(), h4.begin())) );
    stable_sort_by_key( d1.begin(),
                        d1.end(),
                        make_zip_iterator(make_tuple(d3.begin(), d4.begin())) );
    
    // sort with (tuple, tuple)
    stable_sort_by_key( make_zip_iterator(make_tuple(h1.begin(), h2.begin())),
                        make_zip_iterator(make_tuple(h1.end(),   h2.end())),
                        make_zip_iterator(make_tuple(h3.begin(), h4.begin())) );
    stable_sort_by_key( make_zip_iterator(make_tuple(d1.begin(), d2.begin())),
                        make_zip_iterator(make_tuple(d1.end(),   d2.end())),
                        make_zip_iterator(make_tuple(d3.begin(), d4.begin())) );
  
    ASSERT_EQUAL_QUIET(h1, d1);
    ASSERT_EQUAL_QUIET(h2, d2);
    ASSERT_EQUAL_QUIET(h3, d3);
    ASSERT_EQUAL_QUIET(h4, d4);
  }
};
VariableUnitTest<TestZipIteratorStableSortByKey, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestZipIteratorStableSortByKeyInstance;

