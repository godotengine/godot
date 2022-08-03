#include <unittest/unittest.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>

using namespace unittest;

template<typename T>
  struct TestZipIteratorManipulation
{
  template<typename Vector>
  void test(void)
  {
    using namespace thrust;

    Vector v0(4);
    Vector v1(4);
    Vector v2(4);

    // initialize input
    sequence(v0.begin(), v0.end());
    sequence(v1.begin(), v1.end());
    sequence(v2.begin(), v2.end());

    typedef tuple<typename Vector::iterator, typename Vector::iterator> IteratorTuple;

    IteratorTuple t = make_tuple(v0.begin(), v1.begin());

    typedef zip_iterator<IteratorTuple> ZipIterator;

    // test construction
    ZipIterator iter0 = make_zip_iterator(t);

    ASSERT_EQUAL_QUIET(v0.begin(), get<0>(iter0.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin(), get<1>(iter0.get_iterator_tuple()));

    // test dereference
    ASSERT_EQUAL(*v0.begin(), get<0>(*iter0));
    ASSERT_EQUAL(*v1.begin(), get<1>(*iter0));

    // test equality
    ZipIterator iter1 = iter0;
    ZipIterator iter2 = make_zip_iterator(make_tuple(v0.begin(), v2.begin()));
    ZipIterator iter3 = make_zip_iterator(make_tuple(v1.begin(), v2.begin()));
    ASSERT_EQUAL(true,  iter0 == iter1);
    ASSERT_EQUAL(true,  iter0 == iter2);
    ASSERT_EQUAL(false, iter0 == iter3);

    // test inequality
    ASSERT_EQUAL(false, iter0 != iter1);
    ASSERT_EQUAL(false, iter0 != iter2);
    ASSERT_EQUAL(true,  iter0 != iter3);

    // test advance
    ZipIterator iter4 = iter0 + 1;
    ASSERT_EQUAL_QUIET(v0.begin() + 1, get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 1, get<1>(iter4.get_iterator_tuple()));

    // test pre-increment
    ++iter4;
    ASSERT_EQUAL_QUIET(v0.begin() + 2, get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 2, get<1>(iter4.get_iterator_tuple()));

    // test post-increment
    iter4++;
    ASSERT_EQUAL_QUIET(v0.begin() + 3, get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 3, get<1>(iter4.get_iterator_tuple()));

    // test pre-decrement
    --iter4;
    ASSERT_EQUAL_QUIET(v0.begin() + 2, get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 2, get<1>(iter4.get_iterator_tuple()));

    // test post-decrement
    iter4--;
    ASSERT_EQUAL_QUIET(v0.begin() + 1, get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 1, get<1>(iter4.get_iterator_tuple()));

    // test difference
    ASSERT_EQUAL( 1, iter4 - iter0);
    ASSERT_EQUAL(-1, iter0 - iter4);
  }

  void operator()(void)
  {
    test<   thrust::host_vector<T> >();
    test< thrust::device_vector<T> >();
  }
};
SimpleUnitTest<TestZipIteratorManipulation, type_list<int> > TestZipIteratorManipulationInstance;

template <typename T>
  struct TestZipIteratorReference
{
  void operator()(void)
  {
    using namespace thrust;

    // test host types
    typedef typename host_vector<T>::iterator          Iterator1;
    typedef typename host_vector<T>::const_iterator    Iterator2;
    typedef tuple<Iterator1,Iterator2>                 IteratorTuple1;
    typedef zip_iterator<IteratorTuple1> ZipIterator1;

    typedef typename iterator_reference<ZipIterator1>::type zip_iterator_reference_type1;

    host_vector<T> h_variable(1);

    typedef tuple<T&,const T&> reference_type1;

    reference_type1               ref1(*h_variable.begin(),*h_variable.cbegin());
    zip_iterator_reference_type1 test1(*h_variable.begin(),*h_variable.cbegin());

    ASSERT_EQUAL_QUIET(ref1, test1);
    ASSERT_EQUAL( get<0>(ref1),  get<0>(test1));
    ASSERT_EQUAL( get<1>(ref1),  get<1>(test1));


    // test device types
    typedef typename device_vector<T>::iterator        Iterator3;
    typedef typename device_vector<T>::const_iterator  Iterator4;
    typedef tuple<Iterator3,Iterator4>                 IteratorTuple2;
    typedef zip_iterator<IteratorTuple2> ZipIterator2;

    typedef typename iterator_reference<ZipIterator2>::type zip_iterator_reference_type2;

    device_vector<T> d_variable(1);

    typedef tuple< device_reference<T>, device_reference<const T> > reference_type2;

    reference_type2               ref2(*d_variable.begin(),*d_variable.cbegin());
    zip_iterator_reference_type2 test2(*d_variable.begin(),*d_variable.cbegin());

    ASSERT_EQUAL_QUIET(ref2, test2);
    ASSERT_EQUAL( get<0>(ref2),  get<0>(test2));
    ASSERT_EQUAL( get<1>(ref2),  get<1>(test2));
  } // end operator()()
};
SimpleUnitTest<TestZipIteratorReference, NumericTypes> TestZipIteratorReferenceInstance;


template <typename T>
  struct TestZipIteratorTraversal
{
  void operator()(void)
  {
    using namespace thrust;

#if 0
    // test host types
    typedef typename host_vector<T>::iterator          Iterator1;
    typedef typename host_vector<T>::const_iterator    Iterator2;
    typedef tuple<Iterator1,Iterator2>                 IteratorTuple1;
    typedef zip_iterator<IteratorTuple1> ZipIterator1;

    typedef typename iterator_traversal<ZipIterator1>::type zip_iterator_traversal_type1;
#endif

    //ASSERT_EQUAL(true, (detail::is_convertible<zip_iterator_traversal_type1, random_access_traversal_tag>::value) );


#if 0
    // test device types
    typedef typename device_vector<T>::iterator        Iterator3;
    typedef typename device_vector<T>::const_iterator  Iterator4;
    typedef tuple<Iterator3,Iterator4>                 IteratorTuple2;
    typedef zip_iterator<IteratorTuple2> ZipIterator2;

    typedef typename iterator_traversal<ZipIterator2>::type zip_iterator_traversal_type2;
#endif

    //ASSERT_EQUAL(true, (detail::is_convertible<zip_iterator_traversal_type2, thrust::random_access_traversal_tag>::value) );
  } // end operator()()
};
SimpleUnitTest<TestZipIteratorTraversal, NumericTypes> TestZipIteratorTraversalInstance;


template <typename T>
  struct TestZipIteratorSystem
{
  void operator()(void)
  {
    using namespace thrust;

    // XXX these assertions complain about undefined references to integral_constant<...>::value

#if 0
    // test host types
    typedef typename host_vector<T>::iterator          Iterator1;
    typedef typename host_vector<T>::const_iterator    Iterator2;
    typedef tuple<Iterator1,Iterator2>                 IteratorTuple1;
    typedef zip_iterator<IteratorTuple1> ZipIterator1;

    typedef typename iterator_system<ZipIterator1>::type zip_iterator_system_type1;
#endif

    //ASSERT_EQUAL(true, (detail::is_same<zip_iterator_system_type1, experimental::space::host>::value) );


#if 0
    // test device types
    typedef typename device_vector<T>::iterator        Iterator3;
    typedef typename device_vector<T>::const_iterator  Iterator4;
    typedef tuple<Iterator3,Iterator4>                 IteratorTuple2;
    typedef zip_iterator<IteratorTuple1> ZipIterator2;

    typedef typename iterator_system<ZipIterator2>::type zip_iterator_system_type2;
#endif

    //ASSERT_EQUAL(true, (detail::is_convertible<zip_iterator_system_type2, experimental::space::device>::value) );


#if 0
    // test any
    typedef counting_iterator<T>         Iterator5;
    typedef counting_iterator<const T>   Iterator6;
    typedef tuple<Iterator5, Iterator6>                IteratorTuple3;
    typedef zip_iterator<IteratorTuple3> ZipIterator3;

    typedef typename iterator_system<ZipIterator3>::type zip_iterator_system_type3;
#endif

    //ASSERT_EQUAL(true, (detail::is_convertible<zip_iterator_system_type3, thrust::experimental::space::any>::value) );

    
#if 0
    // test host/any
    typedef tuple<Iterator1, Iterator5>                IteratorTuple4;
    typedef zip_iterator<IteratorTuple4> ZipIterator4;

    typedef typename iterator_system<ZipIterator4>::type zip_iterator_system_type4;
#endif

    //ASSERT_EQUAL(true, (detail::is_convertible<zip_iterator_system_type4, thrust::host_system_tag>::value) );


#if 0
    // test any/host
    typedef tuple<Iterator5, Iterator1>                IteratorTuple5;
    typedef zip_iterator<IteratorTuple5> ZipIterator5;

    typedef typename iterator_system<ZipIterator5>::type zip_iterator_system_type5;
#endif

    //ASSERT_EQUAL(true, (detail::is_convertible<zip_iterator_system_type5, thrust::host_system_tag>::value) );


#if 0
    // test device/any
    typedef tuple<Iterator3, Iterator5>                IteratorTuple6;
    typedef zip_iterator<IteratorTuple6> ZipIterator6;

    typedef typename iterator_system<ZipIterator6>::type zip_iterator_system_type6;
#endif

    //ASSERT_EQUAL(true, (detail::is_convertible<zip_iterator_system_type6, thrust::device_system_tag>::value) );


#if 0
    // test any/device
    typedef tuple<Iterator5, Iterator3>                IteratorTuple7;
    typedef zip_iterator<IteratorTuple7> ZipIterator7;

    typedef typename iterator_system<ZipIterator7>::type zip_iterator_system_type7;
#endif

    //ASSERT_EQUAL(true, (detail::is_convertible<zip_iterator_system_type7, thrust::device_system_tag>::value) );
  } // end operator()()
};
SimpleUnitTest<TestZipIteratorSystem, NumericTypes> TestZipIteratorSystemInstance;


template <typename Vector>
void TestZipIteratorCopy(void)
{
  using namespace thrust;
  using T = typename Vector::value_type;

  Vector input0(4),  input1(4);
  Vector output0(4), output1(4);

  // initialize input
  sequence(input0.begin(), input0.end(), T{0});
  sequence(input1.begin(), input1.end(), T{13});

  copy( make_zip_iterator(make_tuple(input0.begin(),  input1.begin())),
        make_zip_iterator(make_tuple(input0.end(),    input1.end())),
        make_zip_iterator(make_tuple(output0.begin(), output1.begin())));

  ASSERT_EQUAL(input0, output0);
  ASSERT_EQUAL(input1, output1);
}
DECLARE_VECTOR_UNITTEST(TestZipIteratorCopy);


struct SumTwoTuple
{
  template<typename Tuple>
  __host__ __device__
  typename thrust::detail::remove_reference<typename thrust::tuple_element<0,Tuple>::type>::type
    operator()(Tuple x) const
  {
    return thrust::get<0>(x) + thrust::get<1>(x);
  }
}; // end SumTwoTuple

struct SumThreeTuple
{
  template<typename Tuple>
  __host__ __device__
  typename thrust::detail::remove_reference<typename thrust::tuple_element<0,Tuple>::type>::type
    operator()(Tuple x) const
  {
    return thrust::get<0>(x) + thrust::get<1>(x) + thrust::get<2>(x);
  }
}; // end SumThreeTuple


template <typename T>
struct TestZipIteratorTransform
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T> h_data0 = unittest::random_samples<T>(n);
    host_vector<T> h_data1 = unittest::random_samples<T>(n);
    host_vector<T> h_data2 = unittest::random_samples<T>(n);

    device_vector<T> d_data0 = h_data0;
    device_vector<T> d_data1 = h_data1;
    device_vector<T> d_data2 = h_data2;

    host_vector<T>   h_result(n);
    device_vector<T> d_result(n);

    // Tuples with 2 elements
    transform( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin())),
               make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end())),
               h_result.begin(),
               SumTwoTuple());
    transform( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin())),
               make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end())),
               d_result.begin(),
               SumTwoTuple());
    ASSERT_EQUAL(h_result, d_result);
    
    
    // Tuples with 3 elements
    transform( make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin(), h_data2.begin())),
               make_zip_iterator(make_tuple(h_data0.end(),   h_data1.end(),   h_data2.end())),
               h_result.begin(),
               SumThreeTuple());
    transform( make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin(), d_data2.begin())),
               make_zip_iterator(make_tuple(d_data0.end(),   d_data1.end(),   d_data2.end())),
               d_result.begin(),
               SumThreeTuple());
    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestZipIteratorTransform, ThirtyTwoBitTypes> TestZipIteratorTransformInstance;


void TestZipIteratorCopyAoSToSoA(void)
{
  using namespace thrust;

  const size_t n = 1;

  typedef tuple<int,int> structure;
  typedef host_vector<structure>   host_array_of_structures;
  typedef device_vector<structure> device_array_of_structures;

  typedef zip_iterator<
    tuple<host_vector<int>::iterator, host_vector<int>::iterator>
  > host_structure_of_arrays;

  typedef zip_iterator<
    tuple<device_vector<int>::iterator, device_vector<int>::iterator>
  > device_structure_of_arrays;

  host_array_of_structures   h_aos(n, make_tuple(7, 13) );
  device_array_of_structures d_aos(n, make_tuple(7, 13) );



  // host to host
  host_vector<int> h_field0(n), h_field1(n);
  host_structure_of_arrays h_soa = make_zip_iterator( make_tuple(h_field0.begin(), h_field1.begin()) );

  thrust::copy(h_aos.begin(), h_aos.end(), h_soa);
  ASSERT_EQUAL_QUIET(make_tuple(7, 13), h_soa[0]);



  // host to device
  device_vector<int> d_field0(n), d_field1(n);
  device_structure_of_arrays d_soa = make_zip_iterator( make_tuple(d_field0.begin(), d_field1.begin()) );

  thrust::copy(h_aos.begin(), h_aos.end(), d_soa);
  ASSERT_EQUAL_QUIET(make_tuple(7, 13), d_soa[0]);



  // device to device
  thrust::fill(d_field0.begin(), d_field0.end(), 0);
  thrust::fill(d_field1.begin(), d_field1.end(), 0);

  thrust::copy(d_aos.begin(), d_aos.end(), d_soa);
  ASSERT_EQUAL_QUIET(make_tuple(7, 13), d_soa[0]);


  // device to host
  thrust::fill(h_field0.begin(), h_field0.end(), 0);
  thrust::fill(h_field1.begin(), h_field1.end(), 0);

  thrust::copy(d_aos.begin(), d_aos.end(), h_soa);
  ASSERT_EQUAL_QUIET(make_tuple(7, 13), h_soa[0]);
};
DECLARE_UNITTEST(TestZipIteratorCopyAoSToSoA);



void TestZipIteratorCopySoAToAoS(void)
{
  using namespace thrust;

  const size_t n = 1;

  typedef tuple<int,int> structure;
  typedef host_vector<structure>   host_array_of_structures;
  typedef device_vector<structure> device_array_of_structures;

  typedef zip_iterator<
    tuple<host_vector<int>::iterator, host_vector<int>::iterator>
  > host_structure_of_arrays;

  typedef zip_iterator<
    tuple<device_vector<int>::iterator, device_vector<int>::iterator>
  > device_structure_of_arrays;

  host_vector<int>   h_field0(n, 7), h_field1(n, 13);
  device_vector<int> d_field0(n, 7), d_field1(n, 13);

  host_structure_of_arrays   h_soa = make_zip_iterator(make_tuple(h_field0.begin(), h_field1.begin()));
  device_structure_of_arrays d_soa = make_zip_iterator(make_tuple(d_field0.begin(), d_field1.begin()));

  host_array_of_structures   h_aos(n);
  device_array_of_structures d_aos(n);



  // host to host
  thrust::fill(h_aos.begin(), h_aos.end(), make_tuple(0,0));

  thrust::copy(h_soa, h_soa + n, h_aos.begin());
  ASSERT_EQUAL_QUIET(7,  get<0>(h_soa[0]));
  ASSERT_EQUAL_QUIET(13, get<1>(h_soa[0]));



  // host to device
  thrust::fill(d_aos.begin(), d_aos.end(), make_tuple(0,0));

  thrust::copy(h_soa, h_soa + n, d_aos.begin());
  ASSERT_EQUAL_QUIET(7,  get<0>(d_soa[0]));
  ASSERT_EQUAL_QUIET(13, get<1>(d_soa[0]));



  // device to device
  thrust::fill(d_aos.begin(), d_aos.end(), make_tuple(0,0));

  thrust::copy(d_soa, d_soa + n, d_aos.begin());
  ASSERT_EQUAL_QUIET(7,  get<0>(d_soa[0]));
  ASSERT_EQUAL_QUIET(13, get<1>(d_soa[0]));



  // device to host
  thrust::fill(h_aos.begin(), h_aos.end(), make_tuple(0,0));

  thrust::copy(d_soa, d_soa + n, h_aos.begin());
  ASSERT_EQUAL_QUIET(7,  get<0>(h_soa[0]));
  ASSERT_EQUAL_QUIET(13, get<1>(h_soa[0]));
};
DECLARE_UNITTEST(TestZipIteratorCopySoAToAoS);

