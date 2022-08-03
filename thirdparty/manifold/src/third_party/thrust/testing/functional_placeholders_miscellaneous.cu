#include <unittest/unittest.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

template<typename T>
  struct saxpy_reference
{
  __host__ __device__ saxpy_reference(const T &aa)
    : a(aa)
  {}

  __host__ __device__ T operator()(const T &x, const T &y) const
  {
    return a * x + y;
  }

  T a;
};

template<typename Vector>
  struct TestFunctionalPlaceholdersValue
{
  void operator()(const size_t)
  {
    const size_t n = 10000;
    typedef typename Vector::value_type T;

    T a(13);

    Vector x = unittest::random_integers<T>(n);
    Vector y = unittest::random_integers<T>(n);
    Vector result(n), reference(n);

    thrust::transform(x.begin(), x.end(), y.begin(), reference.begin(), saxpy_reference<T>(a));

    using namespace thrust::placeholders;
    thrust::transform(x.begin(), x.end(), y.begin(), result.begin(), a * _1 + _2);

    ASSERT_ALMOST_EQUAL(reference, result);
  }
};
VectorUnitTest<TestFunctionalPlaceholdersValue, ThirtyTwoBitTypes, thrust::device_vector, thrust::device_allocator> TestFunctionalPlaceholdersValueDevice;
VectorUnitTest<TestFunctionalPlaceholdersValue, ThirtyTwoBitTypes, thrust::host_vector, std::allocator> TestFunctionalPlaceholdersValueHost;

template<typename Vector>
  struct TestFunctionalPlaceholdersTransformIterator
{
  void operator()(const size_t)
  {
    const size_t n = 10000;
    typedef typename Vector::value_type T;

    T a(13);

    Vector x = unittest::random_integers<T>(n);
    Vector y = unittest::random_integers<T>(n);
    Vector result(n), reference(n);

    thrust::transform(x.begin(), x.end(), y.begin(), reference.begin(), saxpy_reference<T>(a));

    using namespace thrust::placeholders;
    thrust::transform(thrust::make_transform_iterator(x.begin(), a * _1),
                      thrust::make_transform_iterator(x.end(), a * _1),
                      y.begin(),
                      result.begin(),
                      _1 + _2);

    ASSERT_ALMOST_EQUAL(reference, result);
  }
};
VectorUnitTest<TestFunctionalPlaceholdersTransformIterator, ThirtyTwoBitTypes, thrust::device_vector, thrust::device_allocator> TestFunctionalPlaceholdersTransformIteratorInstanceDevice;
VectorUnitTest<TestFunctionalPlaceholdersTransformIterator, ThirtyTwoBitTypes, thrust::host_vector, std::allocator> TestFunctionalPlaceholdersTransformIteratorInstanceHost;

